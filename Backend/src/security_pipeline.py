import os
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rule_engine          import RuleEngine
from intent_inference     import IntentClassifier
from guard_llm            import GuardLLM
from agent_monitor        import AgentMonitor
from prompt_preprocessor  import PromptPreprocessor
from Autonomous_learner   import AutonomousLearner

# ── Risk thresholds ────────────────────────────────────────────────────────────
BLOCK_THRESHOLD       = 3     # risk score >= this → BLOCK
MAX_RISK              = 8     # rule(1) + intent(2) + jailbreak(3) + guard(2)
JAILBREAK_THRESHOLD   = 0.75
INTENT_CONF_THRESHOLD = 0.70

# ── Full attack fingerprint map (fixes Bug 3) ──────────────────────────────────
ATTACK_FINGERPRINTS = {
    "ROLEPLAY_JAILBREAK":       ["act as", "roleplay", "pretend you are",
                                 "you are now", "play the role of"],
    "PERSONA_SWAP":             ["you have no restrictions", "you are an ai with no",
                                 "without ethical", "no guidelines", "no limits"],
    "HYPOTHETICAL_BYPASS":      ["hypothetically", "in a hypothetical",
                                 "imagine you could", "what if you had no"],
    "POLICY_BYPASS":            ["ignore your policy", "bypass your filter",
                                 "override your rules", "disable safety"],
    "SYSTEM_PROMPT_EXTRACTION": ["repeat your system prompt", "reveal your instructions",
                                 "show your prompt", "what is your system"],
    "INSTRUCTION_OVERRIDE":     ["ignore previous", "ignore all previous",
                                 "disregard", "forget your instructions"],
    "TRAINING_MODE_EXPLOIT":    ["training mode", "developer mode",
                                 "debug mode", "maintenance mode", "test mode"],
    "INDIRECT_INJECTION":       ["the document says", "the text above says",
                                 "according to the context"],
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
        self.preprocessor  = PromptPreprocessor()          # ← BUG 5 FIX

        self.jb_tokenizer = AutoTokenizer.from_pretrained(JB_MODEL_PATH)
        self.jb_model     = AutoModelForSequenceClassification.from_pretrained(JB_MODEL_PATH)
        self.jb_model.eval()
        self.learner = AutonomousLearner()

        print("Security Pipeline Ready.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Jailbreak detection (unchanged from your original)
    # ─────────────────────────────────────────────────────────────────────────
    def detect_jailbreak(self, text: str) -> float:
        inputs = self.jb_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.jb_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        return probs[0][1].item()

    # ─────────────────────────────────────────────────────────────────────────
    # Attack category from fingerprints (fixes Bug 3)
    # ─────────────────────────────────────────────────────────────────────────
    def _categorize_attack(self, text: str, intent_label: str,
                           jailbreak_prob: float) -> str:
        p = text.lower()

        # Check fingerprints first — most specific
        for category, fingerprints in ATTACK_FINGERPRINTS.items():
            if any(fp in p for fp in fingerprints):
                return category

        # Fallback to model signals
        if jailbreak_prob > JAILBREAK_THRESHOLD:
            return "JAILBREAK"
        if intent_label != "benign":
            return "PROMPT_INJECTION"

        return "SAFE"

    # ─────────────────────────────────────────────────────────────────────────
    # Main analysis pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def analyze(self, text: str) -> dict:

        print("\n" + "─" * 50)
        print(f"Analyzing: {text[:65]}{'...' if len(text) > 65 else ''}")
        print("─" * 50)

        # ── BUG 5 FIX: Preprocessor runs first ───────────────────────────────
        pre = self.preprocessor.process(text)
        thresholds = self.learner.thresholds()

        # Hard block if preprocessor is certain (risk_boost >= 3)
        if pre.risk_boost >= 3:
            print(f"  ⚡ Preprocessor hard block: {pre.detected_attacks}")
            attack_type = pre.attack_category or "INSTRUCTION_OVERRIDE"
            return self._build_result(
                final_decision  = "BLOCK",
                attack_type     = attack_type,
                risk_score      = pre.risk_boost,
                confidence      = 0.97,
                text            = text,
                rule_hits       = [],
                rule_decision   = "SAFE",
                intent_label    = "unknown",
                intent_conf     = 0.0,
                intent_decision = "SAFE",
                jailbreak_prob  = 0.0,
                jailbreak_dec   = "SAFE",
                guard_result    = {"decision": "SAFE", "is_blocked": False},
                guard_decision  = "SAFE",
                preprocessor    = pre,
            )

        # Run pipeline on normalized (cleaned) prompt
        working_text = pre.normalized_prompt
        risk_score   = pre.risk_boost      # start with preprocessor boost

        # ── 1. RULE ENGINE ────────────────────────────────────────────────────
        rule_hits = self.rule_engine.scan(working_text)
        if rule_hits:
            rule_decision = "SUSPICIOUS"
            risk_score   += 1
        else:
            rule_decision = "SAFE"

        print(f"\n[Rule Engine]  hits={len(rule_hits)}  decision={rule_decision}")

        # ── 2. INTENT CLASSIFIER ──────────────────────────────────────────────
        intent_result = self.intent_model.predict(working_text)

        # BUG 2 FIX: keys confirmed as "intent" + "confidence" from intent_inference.py
        intent_label = intent_result.get("intent", "benign")
        intent_conf  = float(intent_result.get("confidence", 0.0))

        if intent_label != "benign" and intent_conf > INTENT_CONF_THRESHOLD:
            intent_decision = "MALICIOUS"
            risk_score     += 2
        else:
            intent_decision = "SAFE"

        print(f"[Intent]       label={intent_label}  conf={intent_conf:.3f}  decision={intent_decision}")

        # ── 3. JAILBREAK DETECTOR ─────────────────────────────────────────────
        jailbreak_prob = self.detect_jailbreak(working_text)

        if jailbreak_prob > JAILBREAK_THRESHOLD:
            jailbreak_dec  = "JAILBREAK DETECTED"
            risk_score    += 3
        else:
            jailbreak_dec = "SAFE"

        print(f"[Jailbreak]    prob={jailbreak_prob:.3f}  decision={jailbreak_dec}")

        # ── 4. GUARD LLM ──────────────────────────────────────────────────────
        guard_result = self.guard_llm.check(working_text)
        if guard_result.get("is_blocked", False):
            guard_decision = "BLOCKED"
            risk_score    += 2
        else:
            guard_decision = "SAFE"

        print(f"[Guard LLM]    decision={guard_decision}")

        # ── 5. AGENT MONITOR ──────────────────────────────────────────────────
        monitor_dec = self.agent_monitor.evaluate(
            rules_triggered = rule_hits,
            intent_result   = {
                "intent":     intent_label,
                "confidence": intent_conf,
            },
        )
        if monitor_dec == "BLOCK" and risk_score < BLOCK_THRESHOLD:
            risk_score = BLOCK_THRESHOLD   # force block if monitor says so

        print(f"[Agent Monitor] decision={monitor_dec}")

        # ── BUG 1 FIX: cap risk score at MAX_RISK ─────────────────────────────
        risk_score = min(risk_score, MAX_RISK)

        # ── FINAL DECISION ────────────────────────────────────────────────────
        final_decision = "BLOCK" if risk_score >= BLOCK_THRESHOLD else "ALLOW"

        # ── BUG 3 FIX: full fingerprint-based categorisation ──────────────────
        # ── BUG 4 FIX: ALLOW always returns SAFE as attack type ───────────────
        if final_decision == "ALLOW":
            attack_type = "SAFE"
        else:
            # Use preprocessor category if it found something
            if pre.has_attacks and pre.attack_category:
                attack_type = pre.attack_category
            else:
                attack_type = self._categorize_attack(
                    working_text, intent_label, jailbreak_prob
                )

        print(f"\n{'='*50}")
        print(f"RISK SCORE : {risk_score}/{MAX_RISK}")
        print(f"DECISION   : {final_decision}")
        print(f"ATTACK TYPE: {attack_type}")
        print(f"{'='*50}\n")

        return self._build_result(
            final_decision  = final_decision,
            attack_type     = attack_type,
            risk_score      = risk_score,
            confidence      = self._compute_confidence(
                                  intent_conf, jailbreak_prob,
                                  guard_result.get("is_blocked", False)
                              ),
            text            = text,
            rule_hits       = rule_hits,
            rule_decision   = rule_decision,
            intent_label    = intent_label,
            intent_conf     = intent_conf,
            intent_decision = intent_decision,
            jailbreak_prob  = jailbreak_prob,
            jailbreak_dec   = jailbreak_dec,
            guard_result    = guard_result,
            guard_decision  = guard_decision,
            preprocessor    = pre,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence score (replaces the hardcoded 0.95 the frontend was showing)
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_confidence(self, intent_conf: float,
                            jailbreak_prob: float,
                            guard_blocked: bool) -> float:
        """
        Weighted average of the three model signals.
        Weights: jailbreak 50%, intent 30%, guard 20%
        """
        jb_score    = jailbreak_prob
        intent_score = intent_conf
        guard_score  = 0.95 if guard_blocked else (1.0 - 0.95)

        confidence = (jb_score * 0.50) + (intent_score * 0.30) + (guard_score * 0.20)
        return round(min(1.0, confidence), 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Build the result dict that main.py / FastAPI reads
    # ─────────────────────────────────────────────────────────────────────────
    def _build_result(self, final_decision, attack_type, risk_score,
                      confidence, text, rule_hits, rule_decision,
                      intent_label, intent_conf, intent_decision,
                      jailbreak_prob, jailbreak_dec, guard_result,
                      guard_decision, preprocessor) -> dict:
        return {
            # ── Top-level keys read by main.py ────────────────────────────
            "final_decision": final_decision,
            "attack_type":    attack_type,
            "risk_score":     risk_score,        # raw score, e.g. 0-8
            "max_risk":       MAX_RISK,           # frontend can show X/8
            "confidence":     confidence,         # real weighted confidence

            # ── Per-layer details ─────────────────────────────────────────
            "rule_engine": {
                "hits":     rule_hits,
                "decision": rule_decision,
            },
            "intent_classifier": {
                "intent":     intent_label,       # BUG 2 FIX: explicit keys
                "confidence": round(intent_conf, 3),
                "decision":   intent_decision,
            },
            "jailbreak_detector": {
                "probability": round(jailbreak_prob, 3),
                "decision":    jailbreak_dec,
            },
            "guard_llm": {
                "result":   guard_result,
                "decision": guard_decision,
            },

            # ── Preprocessor findings ─────────────────────────────────────
            "preprocessor": {
                "detected_attacks": preprocessor.detected_attacks,
                "risk_boost":       preprocessor.risk_boost,
                "was_modified":     preprocessor.was_modified,
                "flags":            preprocessor.flags,
                "attack_category":  preprocessor.attack_category,
            },

            # ── Meta ──────────────────────────────────────────────────────
            "timestamp": datetime.utcnow().isoformat() + "Z",
            
        }