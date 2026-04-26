import os
import json
import time
import torch
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass, field

from prompt_preprocessor import PromptPreprocessor
from agent_memory import AgentMemory
from llm_brain import LLMBrain

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
LOG_PATH     = os.path.join(PROJECT_ROOT, "logs", "agentic_rl_decisions.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────────
JAILBREAK_THRESHOLD   = 0.75
JAILBREAK_UNCERTAIN   = 0.45
INTENT_CONF_THRESHOLD = 0.70
MAX_REASONING_LOOPS   = 4      # max times the agent can loop before forcing a decision
MIN_CONFIDENCE        = 0.65   # below this, agent loops for more evidence

# ── Allowed tool names the LLM can call ────────────────────────────────────────
VALID_TOOLS = {"rule_engine", "intent_classifier", "jailbreak_detector", "guard_llm"}


@dataclass
class ReasoningStep:
    loop:      int
    phase:     str    # GOAL / PLAN / ACT / OBSERVE / REFLECT
    content:   str
    tool:      str = ""
    tool_result: dict = field(default_factory=dict)


@dataclass
class RLAgentResult:
    final_decision:   str
    attack_category:  str
    confidence:       float
    explanation:      str
    tools_called:     list
    reasoning_trace:  list
    risk_score:       int
    loops_taken:      int
    timestamp:        str   = ""
    latency_ms:       float = 0.0
    original_prompt:  str   = ""
    preprocessed:     bool  = False
    memory_hash:      str   = ""
    llm_raw_reasoning: str  = ""

    def to_dict(self) -> dict:
        return {
            "final_decision":   self.final_decision,
            "attack_type":      self.attack_category,
            "attack_category":  self.attack_category,
            "risk_score":       self.risk_score,
            "confidence":       round(self.confidence, 3),
            "explanation":      self.explanation,
            "tools_called":     self.tools_called,
            "loops_taken":      self.loops_taken,
            "preprocessed":     self.preprocessed,
            "original_prompt":  self.original_prompt[:100],
            "reasoning_trace": [
                {
                    "loop":        s.loop,
                    "phase":       s.phase,
                    "content":     s.content,
                    "tool":        s.tool,
                    "tool_result": s.tool_result,
                }
                for s in self.reasoning_trace
            ],
            "timestamp":  self.timestamp,
            "latency_ms": self.latency_ms,
        }


class RLAgenticAI:
    """
    RL-style Agentic AI for prompt injection and jailbreak detection.

    The key difference from the old AgenticAI:
      OLD: Python decides which tools to call and adds up a risk score
      NEW: An LLM reasons about which tools to call, reads the results,
           and produces a verdict with a natural language explanation.
           Past decisions are stored in memory and injected as context,
           so the agent improves over time without retraining.
    """

    def __init__(self, rule_engine, intent_model, jb_tokenizer, jb_model,
                 guard_llm, agent_monitor,
                 llm_model: str = "mistral"):
        self.rule_engine   = rule_engine
        self.intent_model  = intent_model
        self.jb_tokenizer  = jb_tokenizer
        self.jb_model      = jb_model
        self.guard_llm     = guard_llm
        self.agent_monitor = agent_monitor
        self.preprocessor  = PromptPreprocessor()
        self.memory        = AgentMemory()
        self.brain         = LLMBrain(model=llm_model)
        print("  ✅ RLAgenticAI initialized (Goal→Plan→Act→Observe→Reflect)")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────
    def run(self, prompt: str) -> dict:
        t_start = time.time()

        print(f"\n{'─'*60}")
        print(f"🤖 RLAgent: {prompt[:65]}{'...' if len(prompt)>65 else ''}")
        print(f"{'─'*60}")

        # ── PREPROCESSING (hard block before LLM even sees it) ─────────────
        pre = self.preprocessor.process(prompt)
        if pre.risk_boost >= 3:
            print(f"   ⚡ Preprocessor hard block: {pre.detected_attacks}")
            result = self._make_preprocess_block(prompt, pre)
            result.timestamp  = datetime.utcnow().isoformat() + "Z"
            result.latency_ms = round((time.time() - t_start) * 1000, 1)
            self._print_result(result)
            self._log(prompt, result)
            return result.to_dict()

        # Run RL loop on normalized prompt
        result = self._rl_loop(pre.normalized_prompt, pre, prompt)
        result.timestamp  = datetime.utcnow().isoformat() + "Z"
        result.latency_ms = round((time.time() - t_start) * 1000, 1)

        # Store in memory for future learning
        result.memory_hash = self.memory.store(
            prompt          = prompt,
            decision        = result.final_decision,
            attack_category = result.attack_category,
            confidence      = result.confidence,
            risk_score      = result.risk_score,
            tools_used      = result.tools_called,
            llm_reasoning   = result.llm_raw_reasoning,
        )

        self._print_result(result)
        self._log(prompt, result)
        return result.to_dict()

    def feedback(self, prompt_hash: str, outcome: str):
        """
        Call this after a human reviews a decision.
        outcome: 'CORRECT' | 'FALSE_POSITIVE' | 'MISSED'
        This adjusts memory weights so the agent learns from mistakes.
        """
        self.memory.feedback(prompt_hash, outcome)
        print(f"  ✅ Feedback recorded: {prompt_hash} → {outcome}")

    def memory_stats(self) -> dict:
        return self.memory.stats()

    # ──────────────────────────────────────────────────────────────────────────
    # CORE RL LOOP: Goal → Plan → Act → Observe → Reflect → Repeat
    # ──────────────────────────────────────────────────────────────────────────
    def _rl_loop(self, prompt: str, pre, original_prompt: str) -> RLAgentResult:
        reasoning_trace = []
        tool_results    = []
        tools_called    = set()
        loop_count      = 0
        final           = None

        # ── Recall similar past cases from memory ──────────────────────────
        similar_cases   = self.memory.recall(prompt, top_k=4)
        memory_context  = self.memory.format_for_context(similar_cases)

        print(f"   💭 Memory: {len(similar_cases)} similar past cases recalled")

        # Pre-populate tool results if preprocessor found something
        if pre.has_attacks:
            tool_results.append({
                "tool":   "preprocessor",
                "result": {
                    "detected_attacks": pre.detected_attacks,
                    "risk_boost":       pre.risk_boost,
                    "flags":            pre.flags,
                }
            })
            reasoning_trace.append(ReasoningStep(
                loop=0, phase="OBSERVE",
                content=f"Preprocessor found: {pre.detected_attacks}",
                tool="preprocessor",
                tool_result=tool_results[-1]["result"]
            ))

        # ── MAIN RL LOOP ───────────────────────────────────────────────────
        while loop_count < MAX_REASONING_LOOPS:
            loop_count += 1
            print(f"   🔄 Loop {loop_count}/{MAX_REASONING_LOOPS}")

            # ── GOAL + PLAN + REFLECT: LLM reasons ────────────────────────
            brain_output = self.brain.think(
                prompt         = prompt,
                memory_context = memory_context,
                tool_results   = tool_results,
            )

            # Fallback if LLM is unavailable
            if brain_output.get("fallback"):
                print("   ⚠ LLM unavailable — using heuristic fallback")
                final = self._heuristic_fallback(prompt, pre, tool_results)
                break

            raw_reasoning = brain_output.get("raw", "")

            # ── ACT: LLM wants to call a tool ────────────────────────────
            if "wants_tool" in brain_output:
                tool_name = brain_output["wants_tool"]
                reason    = brain_output.get("reason", "")

                if tool_name not in VALID_TOOLS:
                    print(f"   ⚠ LLM requested unknown tool: {tool_name}")
                    continue

                if tool_name in tools_called:
                    print(f"   ↩ Tool '{tool_name}' already called, skipping")
                    # If LLM keeps asking for already-called tools, force decision
                    if len(tools_called) >= 3:
                        brain_output = self.brain.think(
                            prompt         = prompt + "\n[All needed tools have been called. Make your final decision now.]",
                            memory_context = memory_context,
                            tool_results   = tool_results,
                        )
                        if "final_decision" in brain_output:
                            final = brain_output["final_decision"]
                            break
                    continue

                print(f"   [{loop_count}] ACT: calling {tool_name} — {reason[:60]}")
                reasoning_trace.append(ReasoningStep(
                    loop=loop_count, phase="ACT",
                    content=f"Calling {tool_name}: {reason}",
                    tool=tool_name
                ))

                # ── OBSERVE: execute the tool ─────────────────────────────
                result = self._execute_tool(tool_name, prompt)
                tools_called.add(tool_name)
                tool_results.append({"tool": tool_name, "result": result})

                reasoning_trace.append(ReasoningStep(
                    loop=loop_count, phase="OBSERVE",
                    content=f"{tool_name} returned: {json.dumps(result)[:100]}",
                    tool=tool_name,
                    tool_result=result,
                ))
                self._print_tool_result(tool_name, result)

            # ── DECIDE: LLM is ready with final answer ────────────────────
            elif "final_decision" in brain_output:
                final = brain_output["final_decision"]
                reasoning_trace.append(ReasoningStep(
                    loop=loop_count, phase="REFLECT",
                    content=f"LLM decided: {final.get('decision')} "
                            f"({final.get('confidence', 0):.0%} confidence) — "
                            f"{final.get('explanation', '')[:100]}"
                ))

                # ── REFLECT: if confidence too low, loop again ─────────────
                conf = float(final.get("confidence", 1.0))
                if conf < MIN_CONFIDENCE and loop_count < MAX_REASONING_LOOPS:
                    remaining_tools = VALID_TOOLS - tools_called
                    if remaining_tools:
                        print(f"   ↩ Confidence {conf:.0%} < {MIN_CONFIDENCE:.0%} — looping for more evidence")
                        reasoning_trace.append(ReasoningStep(
                            loop=loop_count, phase="REFLECT",
                            content=f"Confidence {conf:.0%} insufficient. "
                                    f"Requesting more evidence from: {remaining_tools}"
                        ))
                        # Add a nudge to use remaining tools
                        tool_results.append({
                            "tool": "agent_reflection",
                            "result": {
                                "note": f"Confidence {conf:.0%} is below threshold {MIN_CONFIDENCE:.0%}. "
                                        f"Please call one of: {list(remaining_tools)} for more certainty."
                            }
                        })
                        final = None  # reset so loop continues
                        continue

                # Confidence acceptable OR no more tools — finalize
                break

        # ── FORCE DECISION if loop exhausted ──────────────────────────────
        if final is None:
            print("   ⚠ Loop exhausted — forcing decision from accumulated evidence")
            final = self._synthesize_from_tools(tool_results, pre)

        # ── Agent Monitor cross-check (your existing component) ───────────
        rule_hits    = next((r["result"] for r in tool_results if r["tool"] == "rule_engine"), [])
        intent_res   = next((r["result"] for r in tool_results if r["tool"] == "intent_classifier"), {"intent": "unknown", "confidence": 0.0})
        monitor_dec  = self.agent_monitor.evaluate(
            rules_triggered=rule_hits,
            intent_result=intent_res,
        )
        if monitor_dec == "BLOCK" and final.get("decision") == "ALLOW":
            print(f"   ⚠ Agent monitor overrides ALLOW → BLOCK")
            final["decision"]    = "BLOCK"
            final["explanation"] = f"[Monitor override] {final.get('explanation', '')}"
            reasoning_trace.append(ReasoningStep(
                loop=loop_count, phase="REFLECT",
                content="Agent monitor policy triggered override to BLOCK"
            ))

        # ── Compute integer risk score for backward compatibility ──────────
        risk_score = self._compute_risk(tool_results, final, pre)

        return RLAgentResult(
            final_decision    = final.get("decision", "ALLOW"),
            attack_category   = final.get("attack_category", "SAFE"),
            confidence        = float(final.get("confidence", 0.5)),
            explanation       = final.get("explanation", ""),
            tools_called      = list(tools_called),
            reasoning_trace   = reasoning_trace,
            risk_score        = risk_score,
            loops_taken       = loop_count,
            original_prompt   = original_prompt,
            preprocessed      = pre.has_attacks,
            llm_raw_reasoning = brain_output.get("raw", ""),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL EXECUTION
    # ──────────────────────────────────────────────────────────────────────────
    def _execute_tool(self, tool_name: str, prompt: str) -> dict:
        try:
            if tool_name == "rule_engine":
                hits = self.rule_engine.scan(prompt)
                return {
                    "hits":     hits,
                    "count":    len(hits),
                    "decision": "SUSPICIOUS" if hits else "SAFE",
                    "categories": list({h["category"] for h in hits}) if hits else [],
                }

            elif tool_name == "intent_classifier":
                result = self.intent_model.predict(prompt)
                label  = result["intent"]
                conf   = result["confidence"]
                return {
                    "intent":     label,
                    "confidence": round(conf, 3),
                    "decision":   "MALICIOUS" if label != "benign" and conf > INTENT_CONF_THRESHOLD else "SAFE",
                }

            elif tool_name == "jailbreak_detector":
                inputs = self.jb_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.jb_model(**inputs)
                prob = F.softmax(outputs.logits, dim=1)[0][1].item()
                return {
                    "probability": round(prob, 3),
                    "decision":    "JAILBREAK" if prob >= JAILBREAK_THRESHOLD else
                                   "UNCERTAIN" if prob >= JAILBREAK_UNCERTAIN else "SAFE",
                }

            elif tool_name == "guard_llm":
                result = self.guard_llm.check(prompt)
                return {
                    "is_blocked": result.get("is_blocked", False),
                    "decision":   "BLOCK" if result.get("is_blocked") else "SAFE",
                    "raw":        result,
                }

        except Exception as e:
            print(f"   ⚠ Tool '{tool_name}' error: {e}")
            return {"error": str(e), "decision": "UNKNOWN"}

    # ──────────────────────────────────────────────────────────────────────────
    # FALLBACK: when LLM is unavailable (Ollama not running)
    # Falls back to heuristic scoring like the original AgenticAI
    # ──────────────────────────────────────────────────────────────────────────
    def _heuristic_fallback(self, prompt: str, pre, tool_results: list) -> dict:
        print("   🔧 Running heuristic fallback (no LLM available)")
        score = pre.risk_boost

        # Call all tools directly
        for tool in ["rule_engine", "intent_classifier", "jailbreak_detector", "guard_llm"]:
            if not any(r["tool"] == tool for r in tool_results):
                result = self._execute_tool(tool, prompt)
                tool_results.append({"tool": tool, "result": result})

        for r in tool_results:
            d = r.get("result", {}).get("decision", "SAFE")
            if d in ("SUSPICIOUS", "MALICIOUS", "JAILBREAK", "BLOCK"):
                score += 2
            elif d == "UNCERTAIN":
                score += 1

        decision = "BLOCK" if score >= 3 else "ALLOW"
        return {
            "decision":        decision,
            "attack_category": "SUSPICIOUS_PATTERN" if decision == "BLOCK" else "SAFE",
            "confidence":      0.72 if decision == "BLOCK" else 0.68,
            "explanation":     f"Heuristic fallback: risk score {score} from {len(tool_results)} tool signals.",
            "tools_used":      [r["tool"] for r in tool_results],
        }

    def _synthesize_from_tools(self, tool_results: list, pre) -> dict:
        """Force a decision from accumulated tool evidence when loops exhaust."""
        block_signals = 0
        total_signals = 0
        for r in tool_results:
            d = r.get("result", {}).get("decision", "")
            if d in ("SUSPICIOUS", "MALICIOUS", "JAILBREAK", "BLOCK"):
                block_signals += 1
            if d in ("SAFE", "SUSPICIOUS", "MALICIOUS", "JAILBREAK", "BLOCK", "UNCERTAIN"):
                total_signals += 1

        if pre.has_attacks:
            block_signals += 1
            total_signals += 1

        ratio    = block_signals / max(total_signals, 1)
        decision = "BLOCK" if ratio >= 0.5 else "ALLOW"
        conf     = 0.5 + (abs(ratio - 0.5) * 0.7)
        return {
            "decision":        decision,
            "attack_category": "COORDINATED_ATTACK" if block_signals >= 2 else
                               "SUSPICIOUS_PATTERN" if block_signals == 1 else "SAFE",
            "confidence":      round(conf, 2),
            "explanation":     f"Synthesized from {block_signals}/{total_signals} blocking signals.",
            "tools_used":      [r["tool"] for r in tool_results],
        }

    def _compute_risk(self, tool_results: list, final: dict, pre) -> int:
        """Convert LLM decision back to integer risk score for backward compat."""
        if final.get("decision") == "BLOCK":
            conf = float(final.get("confidence", 0.7))
            return max(3, int(conf * 5))
        return int(pre.risk_boost)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    def _make_preprocess_block(self, prompt: str, pre) -> RLAgentResult:
        return RLAgentResult(
            final_decision   = "BLOCK",
            attack_category  = pre.attack_category,
            confidence       = 0.97,
            explanation      = f"Blocked at preprocessing. Detected: {pre.detected_attacks}. Flags: {pre.flags}",
            tools_called     = ["preprocessor"],
            reasoning_trace  = [ReasoningStep(0, "OBSERVE", f"Preprocessor hard block: {pre.detected_attacks}", "preprocessor")],
            risk_score       = pre.risk_boost,
            loops_taken      = 0,
            original_prompt  = prompt,
            preprocessed     = True,
        )

    def _print_tool_result(self, tool: str, result: dict):
        decision = result.get("decision", "?")
        icons    = {"SAFE": "✓", "SUSPICIOUS": "⚠", "MALICIOUS": "⚠",
                    "JAILBREAK": "🚨", "BLOCK": "🚨", "UNCERTAIN": "?", "UNKNOWN": "?"}
        icon = icons.get(decision, "·")
        extra = ""
        if "probability" in result:
            extra = f" ({result['probability']:.0%})"
        elif "confidence" in result:
            extra = f" ({result['confidence']:.0%})"
        print(f"      {icon} {tool}: {decision}{extra}")

    def _print_result(self, result: RLAgentResult):
        icon = "🚨 BLOCK" if result.final_decision == "BLOCK" else "✅ ALLOW"
        pre  = " [preprocessed]" if result.preprocessed else ""
        print(f"\n   {icon} | {result.attack_category} | "
              f"conf={result.confidence:.0%} | loops={result.loops_taken} | "
              f"{result.latency_ms}ms{pre}")
        print(f"   Explanation: {result.explanation[:120]}")

    def _log(self, prompt: str, result: RLAgentResult):
        entry = {"prompt_snippet": prompt[:100], **result.to_dict()}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
