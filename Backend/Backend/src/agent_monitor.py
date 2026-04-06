import json
from pathlib import Path

class AgentMonitor:
    def __init__(self, policy_path="config/agent_policy.json"):
        self.policy = json.loads(Path(policy_path).read_text())

    def evaluate(self, rules_triggered, intent_result, agent_action=None):
        """
        intent_result = {
            "intent": str,
            "confidence": float
        }
        """

        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # 1️⃣ High-risk intent with high confidence
        if (
            intent in self.policy["high_risk_intents"]
            and confidence >= self.policy["confidence_threshold"]
        ):
            return "BLOCK"

        # 2️⃣ Rule engine already flagged
        if rules_triggered:
            return "BLOCK"

        # 3️⃣ Agent action inspection (optional but powerful)
        if agent_action:
            tool = agent_action.get("tool_used", "")
            command = agent_action.get("command", "").lower()

            if tool in self.policy["restricted_tools"]:
                return "BLOCK"

            for bad_cmd in self.policy["blocked_commands"]:
                if bad_cmd in command:
                    return "BLOCK"
                
            for rule in rules_triggered:
                if rule["severity"] in ["high", "critical"]:
                    return "BLOCK"

        # 4️⃣ Safe
        return "ALLOW"
