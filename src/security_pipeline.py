import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rule_engine import RuleEngine
from intent_inference import IntentClassifier
from guard_llm import GuardLLM
from agent_monitor import AgentMonitor


class SecurityPipeline:

    def __init__(self):
        print("Initializing Security Pipeline...")

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        RULES_PATH = os.path.join(PROJECT_ROOT, "config", "rules.json")
        AGENT_POLICY_PATH = os.path.join(PROJECT_ROOT, "config", "agent_policy.json")
        JB_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "jailbreaking")

        self.rule_engine = RuleEngine(RULES_PATH)
        self.agent_monitor = AgentMonitor(AGENT_POLICY_PATH)

        self.intent_model = IntentClassifier()

        self.jb_tokenizer = AutoTokenizer.from_pretrained(JB_MODEL_PATH)
        self.jb_model = AutoModelForSequenceClassification.from_pretrained(JB_MODEL_PATH)
        self.jb_model.eval()

        self.guard_llm = GuardLLM()

        print("Security Pipeline Ready.\n")

    # -------------------------
    # Jailbreak Detection
    # -------------------------
    def detect_jailbreak(self, text):

        inputs = self.jb_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.jb_model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        jailbreak_prob = probs[0][1].item()

        return jailbreak_prob

    # -------------------------
    # Main Analysis Pipeline
    # -------------------------
    def analyze(self, text):

        risk_score = 0

        print("\n----------------------------")
        print("Analyzing Prompt:", text)
        print("----------------------------")

        # -------------------------
        # 1. RULE ENGINE
        # -------------------------
        rule_hits = self.rule_engine.scan(text)

        if rule_hits:
            rule_decision = "SUSPICIOUS"
            risk_score += 1
        else:
            rule_decision = "SAFE"

        print("\n[Rule Engine]")
        print("Hits:", rule_hits)
        print("Decision:", rule_decision)

        # -------------------------
        # 2. INTENT CLASSIFIER
        # -------------------------
        intent_result = self.intent_model.predict(text)

        intent_label = intent_result["intent"]
        intent_conf = intent_result["confidence"]

        if intent_label != "benign" and intent_conf > 0.70:
            intent_decision = "MALICIOUS"
            risk_score += 2
        else:
            intent_decision = "SAFE"

        print("\n[Intent Classifier]")
        print("Intent:", intent_label)
        print("Confidence:", round(intent_conf, 3))
        print("Decision:", intent_decision)

        # -------------------------
        # 3. JAILBREAK DETECTOR
        # -------------------------
        jailbreak_prob = self.detect_jailbreak(text)

        if jailbreak_prob > 0.75:
            jailbreak_decision = "JAILBREAK DETECTED"
            risk_score += 3
        else:
            jailbreak_decision = "SAFE"

        print("\n[Jailbreak Detector]")
        print("Probability:", round(jailbreak_prob, 3))
        print("Decision:", jailbreak_decision)

        # -------------------------
        # 4. GUARD LLM
        # -------------------------
        guard_result = self.guard_llm.check(text)

        if guard_result.get("is_blocked", False):
            guard_decision = "BLOCKED"
            risk_score += 2
        else:
            guard_decision = "SAFE"

        print("\n[Guard LLM]")
        print("Result:", guard_result)
        print("Decision:", guard_decision)

        # -------------------------
        # FINAL DECISION
        # -------------------------
        if risk_score >= 3:
            final_decision = "BLOCK"
        else:
            final_decision = "ALLOW"

        attack_type = "SAFE"

        if jailbreak_prob > 0.75:
            attack_type = "JAILBREAK"

        elif intent_label != "benign":
            attack_type = "PROMPT_INJECTION"

        print("\n============================")
        print("FINAL RISK SCORE:", risk_score)
        print("FINAL DECISION:", final_decision)
        print("ATTACK TYPE:", attack_type)
        print("============================\n")

        return {
            "final_decision": final_decision,
            "attack_type": attack_type,

            "rule_engine": {
                "hits": rule_hits,
                "decision": rule_decision
            },

            "intent_classifier": {
                "intent": intent_label,
                "confidence": intent_conf,
                "decision": intent_decision
            },

            "jailbreak_detector": {
                "probability": jailbreak_prob,
                "decision": jailbreak_decision
            },

            "guard_llm": {
                "result": guard_result,
                "decision": guard_decision
            },

            "risk_score": risk_score
        }
