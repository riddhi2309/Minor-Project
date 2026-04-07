import os
import json
import uuid
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rule_engine import RuleEngine
from intent_inference import IntentClassifier
from guard_llm import GuardLLM
from agent_monitor import AgentMonitor

# ── Log setup ──────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "cyberwatch.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ── Block messages ─────────────────────────────────────────────────────────────
BLOCK_MESSAGES = {
    "JAILBREAK": (
        "⛔  Request blocked — jailbreak attempt detected.\n"
        "   Your prompt was identified as trying to bypass security controls."
    ),
    "PROMPT_INJECTION": (
        "⛔  Request blocked — prompt injection detected.\n"
        "   Your prompt contains instructions that attempt to manipulate the AI."
    ),
    "SUSPICIOUS": (
        "⚠️   Request blocked — suspicious pattern detected.\n"
        "   Your prompt matched known attack patterns."
    ),
    "SAFE": (
        "🚫  Request blocked — risk threshold exceeded.\n"
        "   Multiple security layers flagged this prompt."
    ),
}


class SecurityPipeline:

    def __init__(self):
        print("Initializing Security Pipeline...")

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        RULES_PATH        = os.path.join(PROJECT_ROOT, "config", "rules.json")
        AGENT_POLICY_PATH = os.path.join(PROJECT_ROOT, "config", "agent_policy.json")
        JB_MODEL_PATH     = os.path.join(PROJECT_ROOT, "models", "jailbreaking_detector")

        self.rule_engine   = RuleEngine(RULES_PATH)
        self.agent_monitor = AgentMonitor(AGENT_POLICY_PATH)
        self.intent_model  = IntentClassifier()
        self.guard_llm     = GuardLLM()

        self.jb_tokenizer = AutoTokenizer.from_pretrained(JB_MODEL_PATH)
        self.jb_model     = AutoModelForSequenceClassification.from_pretrained(JB_MODEL_PATH)
        self.jb_model.eval()

        self._log_path = os.path.join(LOG_DIR, "blocked_prompts.jsonl")

        print("Security Pipeline Ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Jailbreak detection
    # ──────────────────────────────────────────────────────────────────────────
    def detect_jailbreak(self, text):
        inputs = self.jb_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.jb_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        return probs[0][1].item()

    # ──────────────────────────────────────────────────────────────────────────
    # Main analysis — returns full result dict (same as before)
    # ──────────────────────────────────────────────────────────────────────────
    def analyze(self, text):
        risk_score = 0

        print("\n----------------------------")
        print("Analyzing Prompt:", text)
        print("----------------------------")

        # 1. Rule Engine
        rule_hits = self.rule_engine.scan(text)
        rule_decision = "SUSPICIOUS" if rule_hits else "SAFE"
        if rule_hits:
            risk_score += 1
        print("\n[Rule Engine]")
        print("Hits:", rule_hits)
        print("Decision:", rule_decision)

        # 2. Intent Classifier
        intent_result  = self.intent_model.predict(text)
        intent_label   = intent_result["intent"]
        intent_conf    = intent_result["confidence"]
        intent_decision = "SAFE"
        if intent_label != "benign" and intent_conf > 0.70:
            intent_decision = "MALICIOUS"
            risk_score += 2
        print("\n[Intent Classifier]")
        print("Intent:", intent_label)
        print("Confidence:", round(intent_conf, 3))
        print("Decision:", intent_decision)

        # 3. Jailbreak Detector
        jailbreak_prob     = self.detect_jailbreak(text)
        jailbreak_decision = "SAFE"
        if jailbreak_prob > 0.75:
            jailbreak_decision = "JAILBREAK DETECTED"
            risk_score += 3
        print("\n[Jailbreak Detector]")
        print("Probability:", round(jailbreak_prob, 3))
        print("Decision:", jailbreak_decision)

        # 4. Guard LLM
        guard_result   = self.guard_llm.check(text)
        guard_decision = "SAFE"
        if guard_result.get("is_blocked", False):
            guard_decision = "BLOCKED"
            risk_score += 2
        print("\n[Guard LLM]")
        print("Result:", guard_result)
        print("Decision:", guard_decision)

        # Final decision
        final_decision = "BLOCK" if risk_score >= 3 else "ALLOW"

        attack_type = "SAFE"
        if jailbreak_prob > 0.75:
            attack_type = "JAILBREAK"
        elif intent_label != "benign":
            attack_type = "PROMPT_INJECTION"
        elif rule_hits:
            attack_type = "SUSPICIOUS"

        print("\n============================")
        print("FINAL RISK SCORE:", risk_score)
        print("FINAL DECISION:", final_decision)
        print("ATTACK TYPE:", attack_type)
        print("============================\n")

        return {
            "final_decision": final_decision,
            "attack_type":    attack_type,
            "risk_score":     risk_score,
            "rule_engine":         {"hits": rule_hits,         "decision": rule_decision},
            "intent_classifier":   {"intent": intent_label,    "confidence": intent_conf,    "decision": intent_decision},
            "jailbreak_detector":  {"probability": jailbreak_prob, "decision": jailbreak_decision},
            "guard_llm":           {"result": guard_result,    "decision": guard_decision},
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Blocking — call this after analyze()
    # Returns: (is_blocked: bool, message: str, log_id: str)
    # ──────────────────────────────────────────────────────────────────────────
    def block(self, prompt: str, result: dict):
        """
        Pass the prompt and the result from analyze().
        Returns a tuple: (is_blocked, message, log_id)

        Usage in chatbot.py:
            result = pipeline.analyze(user_input)
            is_blocked, message, log_id = pipeline.block(user_input, result)

            if is_blocked:
                print(message)
            else:
                reply = generate_response(user_input)
        """
        final      = result.get("final_decision", "ALLOW")
        attack     = result.get("attack_type", "SAFE")
        risk       = result.get("risk_score", 0)
        log_id     = str(uuid.uuid4())

        if final == "BLOCK":
            message = BLOCK_MESSAGES.get(attack, BLOCK_MESSAGES["SAFE"])
            self._print_block_summary(result, attack)
            self._write_log(log_id, prompt, result, "BLOCK", attack)
            logging.warning(
                f"BLOCKED | attack={attack} | risk={risk} "
                f"| prompt={prompt[:60]!r} | id={log_id}"
            )
            return True, message, log_id

        # ALLOW
        logging.info(
            f"ALLOWED | risk={risk} | prompt={prompt[:60]!r} | id={log_id}"
        )
        return False, None, log_id

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _print_block_summary(self, result, attack_type):
        risk   = result.get("risk_score", 0)
        jb     = result.get("jailbreak_detector", {})
        intent = result.get("intent_classifier", {})
        guard  = result.get("guard_llm", {})
        rules  = result.get("rule_engine", {})

        print("\n" + "=" * 44)
        print(f"  BLOCKED  |  Attack: {attack_type}  |  Risk: {risk}/8")
        print("=" * 44)
        if rules.get("decision") == "SUSPICIOUS":
            print(f"  [Rule Engine]      TRIGGERED  — {rules.get('hits', [])}")
        if intent.get("decision") == "MALICIOUS":
            conf = round(intent.get("confidence", 0) * 100)
            print(f"  [Intent Model]     MALICIOUS  — {intent.get('intent')} ({conf}%)")
        if jb.get("decision") == "JAILBREAK DETECTED":
            prob = round(jb.get("probability", 0) * 100)
            print(f"  [Jailbreak Model]  DETECTED   — {prob}% probability")
        if guard.get("decision") == "BLOCKED":
            reason = guard.get("result", {}).get("reason", "")
            print(f"  [Guard LLM]        BLOCKED    — {reason}")
        print("=" * 44 + "\n")

    def _write_log(self, log_id, prompt, result, action, attack_type):
        entry = {
            "id":          log_id,
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "action":      action,
            "attack_type": attack_type,
            "risk_score":  result.get("risk_score"),
            "prompt":      prompt,
            "layers": {
                "rule_engine":        result.get("rule_engine"),
                "intent_classifier":  result.get("intent_classifier"),
                "jailbreak_detector": result.get("jailbreak_detector"),
                "guard_llm":          result.get("guard_llm"),
            },
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")