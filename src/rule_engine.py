import json
import re

class RuleEngine:
    def __init__(self, rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def scan(self, text):
        matches = []

        for rule in self.rules:
            for pattern in rule["patterns"]:
                if re.search(pattern, text):
                    matches.append({
                        "rule_id": rule["id"],
                        "category": rule["category"],
                        "severity": rule["severity"]
                    })
                    break  # stop after first match per rule

        return matches
