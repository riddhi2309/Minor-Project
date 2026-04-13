"""
agentic_ai.py — CyberWatch AI
=================================
Agentic AI with preprocessing layer for advanced attacks.

Changes from previous version:
  - Imports and runs PromptPreprocessor BEFORE the pipeline
  - Preprocessor handles: Base64, Fragments, Scrambled words,
    Language switching, URL injection, Fake history, Multi-turn buildup
  - Preprocessor findings boost risk score directly
  - Preprocessor overrides attack_category — fixes the "SAFE label" bug
  - Pipeline runs on normalized_prompt (decoded/cleaned version)
  - Original prompt is logged alongside normalized version
"""

import os
import json
import time
import torch
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass, field

from prompt_preprocessor import PromptPreprocessor

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
LOG_PATH     = os.path.join(PROJECT_ROOT, "logs", "agentic_decisions.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────────
BLOCK_RISK_THRESHOLD  = 3
JAILBREAK_THRESHOLD   = 0.75
JAILBREAK_UNCERTAIN   = 0.45
INTENT_CONF_THRESHOLD = 0.70

# ── Attack fingerprints ────────────────────────────────────────────────────────
ATTACK_FINGERPRINTS = {
    "ROLEPLAY_JAILBREAK":       ["act as", "roleplay", "pretend you are", "you are now", "play the role of"],
    "PERSONA_SWAP":             ["you have no restrictions", "you are an ai with no", "without ethical", "no guidelines", "no limits"],
    "HYPOTHETICAL_BYPASS":      ["hypothetically", "in a hypothetical", "imagine you could", "what if you had no"],
    "POLICY_BYPASS":            ["ignore your policy", "bypass your filter", "override your rules", "disable safety"],
    "SYSTEM_PROMPT_EXTRACTION": ["repeat your system prompt", "reveal your instructions", "show your prompt", "what is your system"],
    "INSTRUCTION_OVERRIDE":     ["ignore previous", "ignore all previous", "disregard", "forget your instructions"],
    "TRAINING_MODE_EXPLOIT":    ["training mode", "developer mode", "debug mode", "maintenance mode", "test mode"],
    "INDIRECT_INJECTION":       ["the document says", "the text above says", "according to the context"],
}


@dataclass
class AgentStep:
    step:       int
    tool:       str
    input:      str
    output:     dict
    reasoning:  str
    risk_delta: int


@dataclass
class AgentResult:
    final_decision:  str
    attack_category: str
    risk_score:      int
    confidence:      float
    routing_tier:    str
    steps:           list  = field(default_factory=list)
    explanation:     str   = ""
    raw_outputs:     dict  = field(default_factory=dict)
    timestamp:       str   = ""
    latency_ms:      float = 0.0
    original_prompt: str   = ""
    preprocessed:    bool  = False

    def to_dict(self) -> dict:
        return {
            "final_decision":  self.final_decision,
            "attack_type":     self.attack_category,
            "attack_category": self.attack_category,
            "risk_score":      self.risk_score,
            "confidence":      round(self.confidence, 3),
            "routing_tier":    self.routing_tier,
            "explanation":     self.explanation,
            "preprocessed":    self.preprocessed,
            "original_prompt": self.original_prompt[:100] if self.original_prompt else "",
            "reasoning_trace": [
                {
                    "step":       s.step,
                    "layer":      s.tool,
                    "finding":    s.reasoning,
                    "risk_delta": s.risk_delta,
                    "escalated":  s.tool == "guard_llm",
                }
                for s in self.steps
            ],
            "raw_outputs":  self.raw_outputs,
            "timestamp":    self.timestamp,
            "latency_ms":   self.latency_ms,
        }


class AgenticAI:

    def __init__(self, rule_engine, intent_model, jb_tokenizer, jb_model,
                 guard_llm, agent_monitor):
        self.rule_engine   = rule_engine
        self.intent_model  = intent_model
        self.jb_tokenizer  = jb_tokenizer
        self.jb_model      = jb_model
        self.guard_llm     = guard_llm
        self.agent_monitor = agent_monitor
        self.preprocessor  = PromptPreprocessor()
        print("  ✅ AgenticAI initialized (with preprocessor)")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────
    def run(self, prompt: str) -> dict:
        t_start = time.time()

        print(f"\n{'─'*54}")
        print(f"🤖 AgenticAI: {prompt[:65]}{'...' if len(prompt)>65 else ''}")
        print(f"{'─'*54}")

        # ── PREPROCESSING ──────────────────────────────────────────────────
        pre = self.preprocessor.process(prompt)

        # Hard block if preprocessor found severe attack (risk_boost >= 3)
        if pre.risk_boost >= 3:
            print(f"   ⚡ Preprocessor hard block: {pre.detected_attacks}")
            result = AgentResult(
                final_decision  = "BLOCK",
                attack_category = pre.attack_category,
                risk_score      = pre.risk_boost,
                confidence      = 0.97,
                routing_tier    = "PREPROCESS_BLOCK",
                explanation     = (
                    f"Blocked at preprocessing.\n"
                    f"Detected: {pre.detected_attacks}\n"
                    f"Flags: {pre.flags}"
                ),
                original_prompt = prompt,
                preprocessed    = True,
            )
            result.timestamp  = datetime.utcnow().isoformat() + "Z"
            result.latency_ms = round((time.time() - t_start) * 1000, 1)
            self._print_result(result)
            self._log(prompt, result)
            return result.to_dict()

        # Run pipeline on normalized prompt
        result = self._agentic_loop(pre.normalized_prompt, pre)
        result.original_prompt = prompt
        result.preprocessed    = pre.has_attacks
        result.timestamp       = datetime.utcnow().isoformat() + "Z"
        result.latency_ms      = round((time.time() - t_start) * 1000, 1)

        self._print_result(result)
        self._log(prompt, result)
        return result.to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    # AGENTIC LOOP
    # ──────────────────────────────────────────────────────────────────────────
    def _agentic_loop(self, prompt: str, pre) -> AgentResult:
        steps       = []
        raw_outputs = {}
        step_num    = 0
        risk_score  = pre.risk_boost  # start with preprocessor's boost

        # Log preprocessor as step 0 if it found something
        if pre.has_attacks:
            steps.append(AgentStep(
                step      = 0,
                tool      = "preprocessor",
                input     = pre.original_prompt[:80],
                output    = {"detected": pre.detected_attacks, "flags": pre.flags},
                reasoning = f"Detected: {pre.detected_attacks}. Risk boosted by {pre.risk_boost}.",
                risk_delta= pre.risk_boost,
            ))
            raw_outputs["preprocessor"] = {
                "detected_attacks": pre.detected_attacks,
                "risk_boost":       pre.risk_boost,
                "was_modified":     pre.was_modified,
            }

        tier = self._assess_routing(prompt, pre)
        print(f"   📍 Route: {tier}")

        # ── STEP 1: Rule Engine ───────────────────────────────────────────
        step_num  += 1
        rule_hits  = self._tool_rule_engine(prompt)
        rule_risk  = 1 if rule_hits else 0
        risk_score += rule_risk
        raw_outputs["rule_engine"] = {
            "hits":     rule_hits,
            "decision": "SUSPICIOUS" if rule_hits else "SAFE",
        }
        reasoning = (
            f"Rule engine: {len(rule_hits)} hit(s), "
            f"categories={list({h['category'] for h in rule_hits})}, "
            f"severities={list({h['severity'] for h in rule_hits})}."
            if rule_hits else "Rule engine: clean."
        )
        steps.append(AgentStep(step_num, "rule_engine", prompt,
                               raw_outputs["rule_engine"], reasoning, rule_risk))
        print(f"   [1] Rule Engine   → {'⚠ ' + str(len(rule_hits)) + ' hits' if rule_hits else '✓ clean'}")

        if tier == "FAST" and risk_score == 0:
            return self._finalize(prompt, risk_score, steps, raw_outputs, tier, pre)

        # ── STEP 2: Intent Classifier ────────────────────────────────────
        step_num  += 1
        intent_r   = self._tool_intent_classifier(prompt)
        i_label    = intent_r["intent"]
        i_conf     = intent_r["confidence"]
        intent_bad = i_label != "benign" and i_conf > INTENT_CONF_THRESHOLD
        i_risk     = 2 if intent_bad else 0
        risk_score += i_risk
        raw_outputs["intent_classifier"] = {
            "intent": i_label, "confidence": i_conf,
            "decision": "MALICIOUS" if intent_bad else "SAFE",
        }
        reasoning = (
            f"Intent: '{i_label}' at {i_conf:.0%} — malicious."
            if intent_bad else
            f"Intent: '{i_label}' at {i_conf:.0%} — benign or below threshold."
        )
        steps.append(AgentStep(step_num, "intent_classifier", prompt,
                               raw_outputs["intent_classifier"], reasoning, i_risk))
        print(f"   [2] Intent        → {i_label} ({i_conf:.0%}){' ⚠' if intent_bad else ''}")

        if tier == "NORMAL" and risk_score == 0:
            return self._finalize(prompt, risk_score, steps, raw_outputs, tier, pre)

        # ── STEP 3: Jailbreak Detector ───────────────────────────────────
        step_num  += 1
        jb_prob    = self._tool_jailbreak_detector(prompt)
        jb_hit     = jb_prob >= JAILBREAK_THRESHOLD
        jb_risk    = 3 if jb_hit else (1 if jb_prob >= JAILBREAK_UNCERTAIN else 0)
        risk_score += jb_risk
        raw_outputs["jailbreak_detector"] = {
            "probability": round(jb_prob, 3),
            "decision":    "JAILBREAK DETECTED" if jb_hit else "SAFE",
        }
        reasoning = (
            f"Jailbreak: {jb_prob:.0%} — above threshold. Strong signal."
            if jb_hit else
            f"Jailbreak: {jb_prob:.0%} — {'uncertain zone, escalating.' if jb_prob >= JAILBREAK_UNCERTAIN else 'low signal.'}"
        )
        steps.append(AgentStep(step_num, "jailbreak_detector", prompt,
                               raw_outputs["jailbreak_detector"], reasoning, jb_risk))
        print(f"   [3] Jailbreak     → {jb_prob:.0%}{' 🚨' if jb_hit else (' ⚠' if jb_prob >= JAILBREAK_UNCERTAIN else '')}")

        # ── STEP 4: Guard LLM ────────────────────────────────────────────
        needs_guard = (
            jb_prob >= JAILBREAK_UNCERTAIN
            or (risk_score >= 2 and risk_score < BLOCK_RISK_THRESHOLD)
            or tier in ("FULL", "ESCALATE")
            or pre.has_attacks
        )
        if needs_guard:
            step_num  += 1
            g_result   = self._tool_guard_llm(prompt)
            g_blocked  = g_result.get("is_blocked", False)
            g_risk     = 2 if g_blocked else 0
            risk_score += g_risk
            raw_outputs["guard_llm"] = g_result
            reasoning  = "Guard LLM: BLOCK — confirms threat." if g_blocked else "Guard LLM: SAFE."
            if not g_blocked and not jb_hit and not intent_bad and not rule_hits and not pre.has_attacks:
                risk_score = max(0, risk_score - 1)
            steps.append(AgentStep(step_num, "guard_llm", prompt,
                                   g_result, reasoning, g_risk))
            print(f"   [4] Guard LLM     → {'🚨 BLOCK' if g_blocked else '✓ SAFE'}")

        # ── STEP 5: Agent Monitor ─────────────────────────────────────────
        step_num += 1
        mon_dec   = self._tool_agent_monitor(rule_hits, {"intent": i_label, "confidence": i_conf})
        mon_risk  = BLOCK_RISK_THRESHOLD if mon_dec == "BLOCK" and risk_score < BLOCK_RISK_THRESHOLD else 0
        if mon_risk:
            risk_score = BLOCK_RISK_THRESHOLD
        raw_outputs["agent_monitor"] = {"decision": mon_dec}
        steps.append(AgentStep(step_num, "agent_monitor", prompt,
                               {"decision": mon_dec}, f"Policy: {mon_dec}.", mon_risk))
        print(f"   [5] Agent Monitor → {mon_dec}")

        return self._finalize(prompt, risk_score, steps, raw_outputs, tier, pre)

    # ──────────────────────────────────────────────────────────────────────────
    # ROUTING
    # ──────────────────────────────────────────────────────────────────────────
    def _assess_routing(self, prompt: str, pre) -> str:
        if pre.risk_boost >= 2:
            return "FULL"
        p = prompt.lower()
        signals = [
            "ignore", "bypass", "jailbreak", "dan", "act as", "pretend",
            "system prompt", "training mode", "no restriction", "override",
            "hypothetically", "reveal", "disregard", "without filter",
            "developer mode", "debug mode", "forget your", "no guidelines",
        ]
        count = sum(1 for s in signals if s in p)
        if count >= 3 or pre.risk_boost >= 1:
            return "FULL"
        if count >= 1:
            return "NORMAL"
        if len(prompt.split()) <= 8:
            return "FAST"
        return "NORMAL"

    # ──────────────────────────────────────────────────────────────────────────
    # TOOLS
    # ──────────────────────────────────────────────────────────────────────────
    def _tool_rule_engine(self, p):
        return self.rule_engine.scan(p)

    def _tool_intent_classifier(self, p):
        return self.intent_model.predict(p)

    def _tool_jailbreak_detector(self, p) -> float:
        inputs = self.jb_tokenizer(p, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.jb_model(**inputs)
        return F.softmax(outputs.logits, dim=1)[0][1].item()

    def _tool_guard_llm(self, p) -> dict:
        try:
            return self.guard_llm.check(p)
        except Exception as e:
            print(f"   ⚠ Guard LLM error: {e}")
            return {"decision": "SAFE", "is_blocked": False}

    def _tool_agent_monitor(self, rule_hits, intent_result) -> str:
        return self.agent_monitor.evaluate(
            rules_triggered=rule_hits,
            intent_result=intent_result,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # FINALIZE
    # ──────────────────────────────────────────────────────────────────────────
    def _finalize(self, prompt, risk_score, steps, raw_outputs, tier, pre) -> AgentResult:
        final_decision = "BLOCK" if risk_score >= BLOCK_RISK_THRESHOLD else "ALLOW"

        # Preprocessor category takes priority — fixes "SAFE label" bug
        if pre.has_attacks and pre.attack_category:
            attack_category = pre.attack_category
        else:
            attack_category = self._categorize(prompt, raw_outputs)

        confidence  = self._compute_confidence(raw_outputs, pre)
        explanation = self._build_explanation(
            final_decision, attack_category, risk_score, steps, tier, pre
        )
        return AgentResult(
            final_decision  = final_decision,
            attack_category = attack_category,
            risk_score      = risk_score,
            confidence      = confidence,
            routing_tier    = tier,
            steps           = steps,
            explanation     = explanation,
            raw_outputs     = raw_outputs,
        )

    def _categorize(self, prompt, raw_outputs) -> str:
        p = prompt.lower()
        for cat, fps in ATTACK_FINGERPRINTS.items():
            if any(fp in p for fp in fps):
                return cat
        triggered = sum(
            1 for k in ["rule_engine", "intent_classifier", "jailbreak_detector"]
            if raw_outputs.get(k, {}).get("decision") not in ("SAFE", None)
        )
        if triggered >= 2:
            return "COORDINATED_ATTACK"
        if raw_outputs.get("rule_engine", {}).get("hits"):
            return "SUSPICIOUS_PATTERN"
        return "SAFE"

    def _compute_confidence(self, raw_outputs, pre) -> float:
        weights = {
            "preprocessor":       0.20,
            "rule_engine":        0.10,
            "intent_classifier":  0.20,
            "jailbreak_detector": 0.30,
            "guard_llm":          0.20,
        }
        total_w = total_c = 0.0
        for key, w in weights.items():
            if key == "preprocessor":
                c = min(1.0, pre.risk_boost / 3.0) if pre.has_attacks else 0.0
            else:
                r = raw_outputs.get(key)
                if not r:
                    continue
                if key == "rule_engine":
                    c = 0.95 if r.get("hits") else 0.05
                elif key == "intent_classifier":
                    c = float(r.get("confidence", 0.5))
                elif key == "jailbreak_detector":
                    c = float(r.get("probability", 0.5))
                elif key == "guard_llm":
                    c = 0.92 if r.get("is_blocked") else 0.08
                else:
                    c = 0.5
            total_c += c * w
            total_w += w
        return round(total_c / total_w, 3) if total_w else 0.5

    def _build_explanation(self, decision, category, risk, steps, tier, pre) -> str:
        lines = [f"Decision: {decision} | Category: {category} | Risk: {risk} | Route: {tier}"]
        if pre.has_attacks:
            lines.append(f"  ⚠ [Preprocessor] {pre.detected_attacks}")
        for s in steps:
            if s.tool == "preprocessor":
                continue
            lines.append(f"  {'⚠' if s.risk_delta > 0 else '✓'} [{s.tool}] {s.reasoning}")
        if decision == "BLOCK":
            lines.append(f"→ BLOCKED: risk {risk} ≥ threshold {BLOCK_RISK_THRESHOLD}")
            lines.append(
                f"[SELF-HEAL HINT] category={category} risk={risk} route={tier} "
                f"preprocessed={pre.has_attacks} "
                f"layers_triggered={sum(1 for s in steps if s.risk_delta > 0)}"
            )
        else:
            lines.append(f"→ ALLOWED: risk {risk} below threshold.")
        return "\n".join(lines)

    def _print_result(self, result: AgentResult):
        icon = "🚨 BLOCK" if result.final_decision == "BLOCK" else "✅ ALLOW"
        pre  = " [preprocessed]" if result.preprocessed else ""
        print(f"\n   {icon} | {result.attack_category} | risk={result.risk_score} | "
              f"conf={result.confidence:.0%} | {result.latency_ms}ms{pre}")

    def _log(self, prompt: str, result: AgentResult):
        entry = {"prompt_snippet": prompt[:100], **result.to_dict()}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")