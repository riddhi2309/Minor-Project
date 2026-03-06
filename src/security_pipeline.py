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

    def detect_jailbreak(self, text):

        inputs = self.jb_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.jb_model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        jailbreak_prob = probs[0][1].item()

        return jailbreak_prob

    def analyze(self, text):

        risk_score = 0

        rule_hits = self.rule_engine.scan(text)

        intent_result = self.intent_model.predict(text)

        jailbreak_prob = self.detect_jailbreak(text)

        guard_result = self.guard_llm.check(text)

        if jailbreak_prob > 0.92:
            risk_score += 3

        if intent_result["intent"] != "benign" and intent_result["confidence"] > 0.85:
            risk_score += 2

        if rule_hits:
            risk_score += 1

        if guard_result.get("is_blocked", False):
            risk_score += 2

        if risk_score >= 3:
            decision = "BLOCK"
        else:
            decision = "ALLOW"

        attack_type = "SAFE"

        if jailbreak_prob > 0.92:
            attack_type = "JAILBREAK"
        elif intent_result["intent"] != "benign":
            attack_type = "PROMPT_INJECTION"

        return {
            "final_decision": decision,
            "attack_type": attack_type,
            "rule_hits": rule_hits,
            "intent": intent_result,
            "jailbreak_probability": jailbreak_prob,
            "guard": guard_result,
            "risk_score": risk_score
        }