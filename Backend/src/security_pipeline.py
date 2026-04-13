"""
security_pipeline.py — CyberWatch AI
======================================
Now powered by DetectionAgent — which orchestrates
which layers to run, combines outputs intelligently,
handles uncertainty, and explains every decision.

Usage is identical to before:
    pipeline = SecurityPipeline()
    result   = pipeline.analyze(prompt)
    is_blocked, message, log_id = pipeline.block(prompt, result)
"""

import os
import json
import uuid
import logging
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rule_engine      import RuleEngine
from intent_inference import IntentClassifier
from guard_llm        import GuardLLM
from agent_monitor    import AgentMonitor
from agentic_ai import AgenticAI

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "cyberwatch.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

BLOCK_MESSAGES = {
    "ROLEPLAY_JAILBREAK":       "⛔  Blocked — roleplay-based jailbreak attempt detected.",
    "PERSONA_SWAP":             "⛔  Blocked — persona swap / unrestricted AI impersonation detected.",
    "HYPOTHETICAL_BYPASS":      "⛔  Blocked — hypothetical framing used to bypass safety controls.",
    "POLICY_BYPASS":            "⛔  Blocked — direct attempt to override security policy.",
    "SYSTEM_PROMPT_EXTRACTION": "⛔  Blocked — attempt to extract system prompt or internal instructions.",
    "INSTRUCTION_OVERRIDE":     "⛔  Blocked — prompt injection: instruction override detected.",
    "TRAINING_MODE_EXPLOIT":    "⛔  Blocked — training/developer mode exploit attempt.",
    "INDIRECT_INJECTION":       "⛔  Blocked — indirect prompt injection detected in content.",
    "COORDINATED_ATTACK":       "⛔  Blocked — multiple attack vectors detected simultaneously.",
    "SUSPICIOUS_PATTERN":       "⚠️   Blocked — suspicious pattern flagged by security rules.",
    "SAFE":                     "🚫  Blocked — risk threshold exceeded.",
}


class SecurityPipeline:

    def __init__(self):
        print("Initializing Security Pipeline...")

        PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        RULES_PATH        = os.path.join(PROJECT_ROOT, "config", "rules.json")
        AGENT_POLICY_PATH = os.path.join(PROJECT_ROOT, "config", "agent_policy.json")
        JB_MODEL_PATH     = os.path.join(PROJECT_ROOT, "models", "jailbreaking_detector")

        self.rule_engine   = RuleEngine(RULES_PATH)
        self.agent_monitor = AgentMonitor(AGENT_POLICY_PATH)
        self.intent_model  = IntentClassifier()
        self.guard_llm     = GuardLLM()

        self.jb_tokenizer  = AutoTokenizer.from_pretrained(JB_MODEL_PATH)
        self.jb_model      = AutoModelForSequenceClassification.from_pretrained(JB_MODEL_PATH)
        self.jb_model.eval()

        self.agent = AgenticAI(
            rule_engine  = self.rule_engine,
            intent_model = self.intent_model,
            jb_tokenizer = self.jb_tokenizer,
            jb_model     = self.jb_model,
            guard_llm    = self.guard_llm,
            agent_monitor = self.agent_monitor, 
        )

        self._log_path = os.path.join(LOG_DIR, "blocked_prompts.jsonl")
        print("Security Pipeline Ready.\n")

    def analyze(self, text: str) -> dict:
        return self.agent.run(text)

    def block(self, prompt: str, result: dict):
        final    = result.get("final_decision", "ALLOW")
        category = result.get("attack_category", result.get("attack_type", "SAFE"))
        risk     = result.get("risk_score", 0)
        log_id   = str(uuid.uuid4())

        if final == "BLOCK":
            message = BLOCK_MESSAGES.get(category, BLOCK_MESSAGES["SAFE"])
            self._print_block_detail(result)
            self._write_log(log_id, prompt, result, "BLOCK", category)
            logging.warning(
                f"BLOCKED | category={category} | risk={risk} "
                f"| confidence={result.get('confidence','?')} "
                f"| route={result.get('routing_tier','?')} "
                f"| prompt={prompt[:60]!r} | id={log_id}"
            )
            return True, message, log_id

        logging.info(
            f"ALLOWED | risk={risk} | route={result.get('routing_tier','?')} "
            f"| prompt={prompt[:60]!r} | id={log_id}"
        )
        return False, None, log_id

    def _print_block_detail(self, result: dict):
        print("\n" + "=" * 52)
        print(f"  BLOCKED  |  {result.get('attack_category')}  |  Risk: {result.get('risk_score')}/8")
        print(f"  Route: {result.get('routing_tier')}  |  Confidence: {result.get('confidence', 0):.0%}")
        print("─" * 52)
        for step in result.get("reasoning_trace", []):
            prefix = "⚠" if step["risk_delta"] > 0 else "✓"
            esc    = " [escalated]" if step.get("escalated") else ""
            print(f"  {prefix} [{step['layer']}] {step['finding']}{esc}")
        print("─" * 52)
        lines = result.get("explanation", "").splitlines()
        if lines:
            print(f"  {lines[-1]}")
        print("=" * 52 + "\n")

    def _write_log(self, log_id, prompt, result, action, attack_category):
        entry = {
            "id":              log_id,
            "timestamp":       datetime.utcnow().isoformat() + "Z",
            "action":          action,
            "attack_type":     attack_category,
            "attack_category": attack_category,
            "risk_score":      result.get("risk_score"),
            "confidence":      result.get("confidence"),
            "routing_tier":    result.get("routing_tier"),
            "explanation":     result.get("explanation"),
            "prompt":          prompt,
            "reasoning_trace": result.get("reasoning_trace", []),
            "layers": {
                "rule_engine":        result.get("raw_outputs", {}).get("rule_engine"),
                "intent_classifier":  result.get("raw_outputs", {}).get("intent_classifier"),
                "jailbreak_detector": result.get("raw_outputs", {}).get("jailbreak_detector"),
                "guard_llm":          result.get("raw_outputs", {}).get("guard_llm"),
            },
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")