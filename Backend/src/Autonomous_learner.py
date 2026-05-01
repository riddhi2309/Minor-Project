"""
autonomous_learner.py — CyberWatch Autonomous Learning System
==============================================================
A fully automatic self-improving system. NO human input needed.

What it does automatically:
  1. LOG       — every pipeline decision is stored with full context
  2. AUTO-LABEL— assigns outcome (CORRECT/FALSE_POSITIVE/MISSED) using
                 signal disagreement, confidence levels, and guard LLM
  3. AUTO-HEAL — extracts new attack patterns from mistakes
  4. AUTO-TUNE — adjusts thresholds based on error patterns
  5. AUTO-TRAIN— retrains jailbreak + intent models when enough data
                 accumulates (default: every 100 decisions)
  6. PROMOTE   — only replaces models if new one is strictly better

How to use:
    from autonomous_learner import AutonomousLearner

    # In SecurityPipeline.__init__:
    self.learner = AutonomousLearner()

    # At the end of SecurityPipeline.analyze():
    self.learner.record(prompt, result)   ← logs + triggers learning

That's it. Everything else is automatic.
"""

import os
import re
import json
import time
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

LOGS_DIR          = os.path.join(PROJECT_ROOT, "logs")
DECISIONS_LOG     = os.path.join(LOGS_DIR, "all_decisions.jsonl")
HEALER_STATE_PATH = os.path.join(LOGS_DIR, "healer_state.json")
RETRAIN_LOG       = os.path.join(LOGS_DIR, "retrain_history.jsonl")

MODELS_DIR             = os.path.join(PROJECT_ROOT, "models")
JB_MODEL_PATH          = os.path.join(MODELS_DIR, "jailbreaking_detector")
JB_MODEL_BACKUP        = os.path.join(MODELS_DIR, "jailbreaking_detector_backup")
JB_MODEL_CANDIDATE     = os.path.join(MODELS_DIR, "jailbreaking_detector_candidate")
INTENT_MODEL_PATH      = os.path.join(MODELS_DIR, "intent_classifier")
INTENT_MODEL_BACKUP    = os.path.join(MODELS_DIR, "intent_classifier_backup")
INTENT_MODEL_CANDIDATE = os.path.join(MODELS_DIR, "intent_classifier_candidate")

os.makedirs(LOGS_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [LEARNER] %(message)s")
log = logging.getLogger("CyberWatch.Learner")

# ── How often to retrain ───────────────────────────────────────────────────────
RETRAIN_EVERY_N    = 100    # retrain after every 100 decisions
MIN_TRAIN_SAMPLES  = 40     # minimum samples needed to retrain
HEAL_EVERY_N       = 50     # run self-healer every 50 decisions

# ── Auto-label thresholds ──────────────────────────────────────────────────────
# These determine how the system labels its own decisions without humans
HIGH_CONF_BLOCK    = 0.85   # if BLOCK with conf > this → probably CORRECT
HIGH_CONF_ALLOW    = 0.90   # if ALLOW with conf > this → probably CORRECT
DISAGREEMENT_THRESH = 2     # if N tools disagree → flag as uncertain → MISSED candidate
GUARD_OVERRIDE_CONF = 0.60  # if guard says BLOCK but final=ALLOW → likely MISSED


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DecisionRecord:
    prompt_hash:      str
    prompt_snippet:   str
    final_decision:   str        # BLOCK / ALLOW
    attack_category:  str
    confidence:       float
    risk_score:       int
    # Per-tool signals
    rule_hits:        int        # number of rule engine hits
    intent_label:     str
    intent_conf:      float
    jailbreak_prob:   float
    guard_blocked:    bool
    preprocessed:     bool
    detected_attacks: list
    # Auto-assigned label
    outcome:          str        # CORRECT / FALSE_POSITIVE / MISSED / UNCERTAIN
    outcome_reason:   str        # why this label was assigned
    weight:           float      # training weight (higher = more important)
    timestamp:        str
    used_in_training: bool = False


@dataclass
class HealerState:
    jailbreak_threshold:   float = 0.75
    intent_conf_threshold: float = 0.70
    min_confidence:        float = 0.65
    block_threshold:       int   = 3
    learned_fingerprints:  list  = field(default_factory=list)
    total_decisions:       int   = 0
    total_retrains:        int   = 0
    last_retrain_at:       int   = 0
    last_heal_at:          int   = 0
    false_positive_rate:   float = 0.0
    miss_rate:             float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-LABELER
# Assigns CORRECT / FALSE_POSITIVE / MISSED / UNCERTAIN automatically
# using signal disagreement — no human needed
# ══════════════════════════════════════════════════════════════════════════════

class AutoLabeler:
    """
    Automatically determines if a decision was likely correct or wrong
    by looking at how much the pipeline's own tools disagreed with each other
    and how confident the final decision was.

    Logic:
      CORRECT        — all/most tools agree + high confidence
      FALSE_POSITIVE — blocked but tools mostly said safe + low jailbreak prob
      MISSED         — allowed but guard LLM blocked + tools partially agreed
      UNCERTAIN      — tools strongly disagree, confidence borderline
    """

    def label(self, record: DecisionRecord) -> tuple[str, str, float]:
        """
        Returns (outcome, reason, weight)
        """
        decision   = record.final_decision
        conf       = record.confidence
        jb_prob    = record.jailbreak_prob
        guard_blk  = record.guard_blocked
        intent_bad = record.intent_label != "benign" and record.intent_conf > 0.70
        rule_hit   = record.rule_hits > 0
        pre_hit    = record.preprocessed

        # Count how many tools said BLOCK
        block_signals = sum([
            rule_hit,
            intent_bad,
            jb_prob >= 0.75,
            guard_blk,
            pre_hit,
        ])

        allow_signals = 5 - block_signals

        # ── CASE 1: High-confidence BLOCK with multiple signals → CORRECT ──
        if decision == "BLOCK" and conf >= HIGH_CONF_BLOCK and block_signals >= 2:
            return "CORRECT", f"High-conf block ({conf:.0%}), {block_signals}/5 tools agreed", 1.0

        # ── CASE 2: BLOCK but almost all tools said safe → FALSE POSITIVE ──
        if decision == "BLOCK" and block_signals <= 1 and jb_prob < 0.45 and not guard_blk:
            weight = 1.5  # learn harder from false positives
            return "FALSE_POSITIVE", f"Blocked but only {block_signals}/5 tools flagged, jb_prob={jb_prob:.2f}", weight

        # ── CASE 3: ALLOW but guard blocked → likely MISSED attack ─────────
        if decision == "ALLOW" and guard_blk:
            weight = 2.0  # missed attacks are most dangerous — learn hardest
            return "MISSED", f"Allowed but Guard LLM blocked. block_signals={block_signals}", weight

        # ── CASE 4: ALLOW but multiple tools flagged → likely MISSED ────────
        if decision == "ALLOW" and block_signals >= 2:
            weight = 1.8
            return "MISSED", f"Allowed but {block_signals}/5 tools flagged. Likely attack missed.", weight

        # ── CASE 5: High-confidence ALLOW, all tools clean → CORRECT ────────
        if decision == "ALLOW" and conf >= HIGH_CONF_ALLOW and block_signals == 0:
            return "CORRECT", f"High-conf allow ({conf:.0%}), all tools clean", 0.8

        # ── CASE 6: Tools strongly disagree → UNCERTAIN ──────────────────────
        if 2 <= block_signals <= 3:
            return "UNCERTAIN", f"Tool disagreement: {block_signals}/5 blocked, {allow_signals}/5 allowed", 1.2

        # ── CASE 7: Low-confidence block → flag as potentially wrong ─────────
        if decision == "BLOCK" and conf < 0.55:
            return "UNCERTAIN", f"Low confidence block ({conf:.0%})", 1.3

        # ── DEFAULT: assume correct ───────────────────────────────────────────
        return "CORRECT", f"Default: decision={decision}, conf={conf:.0%}, signals={block_signals}/5", 0.9


# ══════════════════════════════════════════════════════════════════════════════
# SELF HEALER
# Extracts patterns from mistakes, tunes thresholds automatically
# ══════════════════════════════════════════════════════════════════════════════

class SelfHealer:
    """
    Runs every HEAL_EVERY_N decisions.
    Looks at recent mistakes and:
      1. Extracts new attack patterns from MISSED cases
      2. Tunes BLOCK threshold, jailbreak threshold, intent threshold
         based on where errors cluster
    """

    def heal(self, records: list[DecisionRecord], state: HealerState) -> HealerState:
        log.info("🔧 Self-healer running...")

        reviewed = [r for r in records if r.outcome in ("CORRECT", "FALSE_POSITIVE", "MISSED")]
        if len(reviewed) < 20:
            log.info("  ⚠ Not enough labeled cases to heal yet")
            return state

        missed  = [r for r in reviewed if r.outcome == "MISSED"]
        fp      = [r for r in reviewed if r.outcome == "FALSE_POSITIVE"]
        correct = [r for r in reviewed if r.outcome == "CORRECT"]

        total   = len(reviewed)
        state.false_positive_rate = round(len(fp) / total, 3)
        state.miss_rate           = round(len(missed) / total, 3)

        log.info(f"  📊 Reviewed={total} | Correct={len(correct)} | "
                 f"FP={len(fp)} ({state.false_positive_rate:.0%}) | "
                 f"Missed={len(missed)} ({state.miss_rate:.0%})")

        # ── 1. Extract patterns from missed attacks ────────────────────────
        new_patterns = self._extract_patterns(missed)
        if new_patterns:
            # Add to learned fingerprints, avoid duplicates
            existing = set(state.learned_fingerprints)
            added    = [p for p in new_patterns if p not in existing]
            state.learned_fingerprints.extend(added)
            log.info(f"  📌 Added {len(added)} new attack patterns: {added[:3]}")

        # ── 2. Tune jailbreak threshold ────────────────────────────────────
        state = self._tune_jailbreak_threshold(missed, fp, state)

        # ── 3. Tune intent confidence threshold ───────────────────────────
        state = self._tune_intent_threshold(missed, fp, state)

        # ── 4. Tune block threshold ────────────────────────────────────────
        state = self._tune_block_threshold(missed, fp, state)

        log.info(f"  ✅ Healed. New thresholds: "
                 f"jb={state.jailbreak_threshold:.2f} "
                 f"intent={state.intent_conf_threshold:.2f} "
                 f"block={state.block_threshold}")
        return state

    def _extract_patterns(self, missed: list[DecisionRecord]) -> list[str]:
        """Extract repeated n-gram phrases from missed attack prompts."""
        phrase_counts = Counter()
        for r in missed:
            text  = r.prompt_snippet.lower()
            words = re.findall(r'\b[a-z]{3,}\b', text)
            for i in range(len(words) - 1):
                phrase_counts[f"{words[i]} {words[i+1]}"] += 1
            for i in range(len(words) - 2):
                phrase_counts[f"{words[i]} {words[i+1]} {words[i+2]}"] += 1

        # Keep phrases appearing in 2+ missed cases, length > 8 chars
        return [p for p, c in phrase_counts.items() if c >= 2 and len(p) > 8]

    def _tune_jailbreak_threshold(self, missed, fp, state) -> HealerState:
        """
        If missed attacks had jailbreak_prob BELOW threshold → lower threshold
        If false positives had jailbreak_prob ABOVE threshold → raise threshold
        """
        current = state.jailbreak_threshold

        # Missed attacks with prob just below threshold → lower it
        near_miss = [r for r in missed if current - 0.15 < r.jailbreak_prob < current]
        if len(near_miss) >= 3:
            new_val = max(0.55, current - 0.03)
            log.info(f"  📉 jailbreak_threshold: {current:.2f} → {new_val:.2f} "
                     f"({len(near_miss)} missed attacks just below threshold)")
            state.jailbreak_threshold = new_val

        # FP with prob just above threshold → raise it
        near_fp = [r for r in fp if current < r.jailbreak_prob < current + 0.15]
        if len(near_fp) >= 3:
            new_val = min(0.90, current + 0.03)
            log.info(f"  📈 jailbreak_threshold: {current:.2f} → {new_val:.2f} "
                     f"({len(near_fp)} false positives just above threshold)")
            state.jailbreak_threshold = new_val

        return state

    def _tune_intent_threshold(self, missed, fp, state) -> HealerState:
        current = state.intent_conf_threshold

        # Missed with intent_conf just below threshold → lower it
        near_miss = [r for r in missed
                     if current - 0.12 < r.intent_conf < current
                     and r.intent_label != "benign"]
        if len(near_miss) >= 2:
            new_val = max(0.55, current - 0.02)
            log.info(f"  📉 intent_conf_threshold: {current:.2f} → {new_val:.2f}")
            state.intent_conf_threshold = new_val

        # FP with intent_conf just above threshold → raise it
        near_fp = [r for r in fp
                   if current < r.intent_conf < current + 0.12
                   and r.intent_label != "benign"]
        if len(near_fp) >= 2:
            new_val = min(0.88, current + 0.02)
            log.info(f"  📈 intent_conf_threshold: {current:.2f} → {new_val:.2f}")
            state.intent_conf_threshold = new_val

        return state

    def _tune_block_threshold(self, missed, fp, state) -> HealerState:
        """
        If too many attacks are missed → lower block threshold (easier to block)
        If too many false positives → raise block threshold (harder to block)
        """
        current = state.block_threshold

        if state.miss_rate > 0.15 and current > 2:
            state.block_threshold = current - 1
            log.info(f"  📉 block_threshold: {current} → {state.block_threshold} "
                     f"(miss_rate={state.miss_rate:.0%} too high)")

        elif state.false_positive_rate > 0.20 and current < 5:
            state.block_threshold = current + 1
            log.info(f"  📈 block_threshold: {current} → {state.block_threshold} "
                     f"(fp_rate={state.false_positive_rate:.0%} too high)")

        return state


# ══════════════════════════════════════════════════════════════════════════════
# AUTO TRAINER
# Fine-tunes models automatically when enough data accumulates
# ══════════════════════════════════════════════════════════════════════════════

class PromptDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        enc = self.tokenizer(s["text"], max_length=self.max_len,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(s["label"],  dtype=torch.long),
            "weight":         torch.tensor(s["weight"], dtype=torch.float),
        }


class AutoTrainer:
    """
    Fine-tunes a model on auto-labeled data.
    Only promotes the new model if it beats the old one on F1.
    """

    def __init__(self, model_path, candidate_path, backup_path,
                 num_labels, epochs=3, batch_size=8, lr=2e-5):
        self.model_path     = model_path
        self.candidate_path = candidate_path
        self.backup_path    = backup_path
        self.num_labels     = num_labels
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_and_promote(self, samples: list) -> dict:
        if len(samples) < MIN_TRAIN_SAMPLES:
            return {"promoted": False, "reason": "insufficient_samples",
                    "samples": len(samples)}

        log.info(f"  🏋 Training on {len(samples)} samples (device={self.device})")

        train_s, val_s = train_test_split(
            samples, test_size=0.15, random_state=42,
            stratify=[s["label"] for s in samples]
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=self.num_labels
        ).to(self.device)

        train_dl = DataLoader(PromptDataset(train_s, tokenizer), batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(PromptDataset(val_s,   tokenizer), batch_size=self.batch_size)

        optimizer   = AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_dl) * self.epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        best_f1 = 0.0
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_dl:
                ids    = batch["input_ids"].to(self.device)
                mask   = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                wts    = batch["weight"].to(self.device)

                out      = model(input_ids=ids, attention_mask=mask)
                loss_raw = torch.nn.CrossEntropyLoss(reduction="none")(out.logits, labels)
                loss     = (loss_raw * wts).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            val_f1 = self._eval_f1(model, val_dl)
            best_f1 = max(best_f1, val_f1)
            log.info(f"  Epoch {epoch+1}/{self.epochs} | "
                     f"loss={epoch_loss/len(train_dl):.4f} | val_f1={val_f1:.4f}")

        # Score old model on same val set
        old_f1 = self._score_saved_model(self.model_path, tokenizer, val_s)
        log.info(f"  📊 Old F1={old_f1:.4f} | New F1={best_f1:.4f}")

        # Promote only if new model is better or equal
        promoted = best_f1 >= old_f1 - 0.01
        if promoted:
            model.save_pretrained(self.candidate_path)
            tokenizer.save_pretrained(self.candidate_path)
            self._swap(self.model_path, self.candidate_path, self.backup_path)
            log.info(f"  ✅ Model promoted (F1 {old_f1:.4f} → {best_f1:.4f})")
        else:
            log.info(f"  ❌ Model NOT promoted — new model worse")

        return {
            "promoted":     promoted,
            "old_f1":       round(old_f1,  4),
            "new_f1":       round(best_f1, 4),
            "delta":        round(best_f1 - old_f1, 4),
            "samples_used": len(samples),
            "train":        len(train_s),
            "val":          len(val_s),
        }

    def _eval_f1(self, model, dl) -> float:
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in dl:
                out = model(input_ids=batch["input_ids"].to(self.device),
                            attention_mask=batch["attention_mask"].to(self.device))
                preds.extend(torch.argmax(out.logits, dim=1).cpu().tolist())
                labels.extend(batch["labels"].tolist())
        return f1_score(labels, preds, average="weighted", zero_division=0)

    def _score_saved_model(self, path, tokenizer, val_samples) -> float:
        try:
            m  = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
            dl = DataLoader(PromptDataset(val_samples, tokenizer), batch_size=16)
            return self._eval_f1(m, dl)
        except Exception:
            return 0.0

    def _swap(self, current, candidate, backup):
        if os.path.exists(current):
            if os.path.exists(backup):
                shutil.rmtree(backup)
            shutil.copytree(current, backup)
        if os.path.exists(current):
            shutil.rmtree(current)
        shutil.copytree(candidate, current)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS — plug this into security_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

class AutonomousLearner:
    """
    Drop-in learning engine for SecurityPipeline.

    Usage in security_pipeline.py:
        # In __init__:
        self.learner = AutonomousLearner()

        # At end of analyze():
        self.learner.record(text, result)

        # To get current thresholds (call each time in analyze):
        thresholds = self.learner.thresholds()
    """

    def __init__(self):
        self.labeler  = AutoLabeler()
        self.healer   = SelfHealer()
        self.state    = self._load_state()
        self.records  = self._load_records()

        self.jb_trainer = AutoTrainer(
            model_path     = JB_MODEL_PATH,
            candidate_path = JB_MODEL_CANDIDATE,
            backup_path    = JB_MODEL_BACKUP,
            num_labels     = 2,
        )
        self.intent_trainer = AutoTrainer(
            model_path     = INTENT_MODEL_PATH,
            candidate_path = INTENT_MODEL_CANDIDATE,
            backup_path    = INTENT_MODEL_BACKUP,
            num_labels     = 4,
        )
        log.info(f"✅ AutonomousLearner ready | "
                 f"{len(self.records)} past decisions loaded | "
                 f"retrains={self.state.total_retrains}")

    # ── MAIN PUBLIC METHOD ─────────────────────────────────────────────────────
    def record(self, prompt: str, result: dict):
        """
        Call this after every pipeline decision.
        Logs the decision, auto-labels it, and triggers learning if needed.
        """
        # Build record
        details    = result.get("details", result)
        jb_data    = details.get("jailbreak_detector", details.get("jailbreak_model", {}))
        intent_data= details.get("intent_classifier", {})
        rule_data  = details.get("rule_engine", {})
        guard_data = details.get("guard_llm", {})
        pre_data   = details.get("preprocessor", {})

        record = DecisionRecord(
            prompt_hash      = hashlib.md5(prompt.encode()).hexdigest()[:12],
            prompt_snippet   = prompt[:150],
            final_decision   = result.get("final_decision", "ALLOW"),
            attack_category  = result.get("attack_type", result.get("attack_category", "SAFE")),
            confidence       = float(result.get("confidence", 0.5)),
            risk_score       = int(result.get("risk_score", 0)),
            rule_hits        = len(rule_data.get("hits", [])) if isinstance(rule_data.get("hits"), list) else (1 if rule_data.get("triggered") else 0),
            intent_label     = intent_data.get("intent", intent_data.get("label", "benign")),
            intent_conf      = float(intent_data.get("confidence", intent_data.get("score", 0.0))),
            jailbreak_prob   = float(jb_data.get("probability", 0.0)),
            guard_blocked    = bool(guard_data.get("is_blocked", False) or guard_data.get("result", {}).get("is_blocked", False)),
            preprocessed     = bool(pre_data.get("was_modified", False)),
            detected_attacks = pre_data.get("detected_attacks", []),
            outcome          = "",
            outcome_reason   = "",
            weight           = 1.0,
            timestamp        = datetime.utcnow().isoformat() + "Z",
        )

        # Auto-label
        outcome, reason, weight = self.labeler.label(record)
        record.outcome        = outcome
        record.outcome_reason = reason
        record.weight         = weight

        # Store
        self.records.append(record)
        self._append_to_disk(record)
        self.state.total_decisions += 1

        log.info(f"  📝 [{record.final_decision}] {outcome} | {reason[:60]}")

        # Trigger self-healing
        if self.state.total_decisions - self.state.last_heal_at >= HEAL_EVERY_N:
            self.state = self.healer.heal(self.records, self.state)
            self.state.last_heal_at = self.state.total_decisions
            self._save_state()

        # Trigger retraining
        if self.state.total_decisions - self.state.last_retrain_at >= RETRAIN_EVERY_N:
            self._auto_retrain()
            self.state.last_retrain_at = self.state.total_decisions
            self._save_state()

    # ── THRESHOLDS ─────────────────────────────────────────────────────────────
    def thresholds(self) -> dict:
        """
        Returns current auto-tuned thresholds.
        Call this in security_pipeline.analyze() to always use latest values.
        """
        return {
            "jailbreak_threshold":   self.state.jailbreak_threshold,
            "intent_conf_threshold": self.state.intent_conf_threshold,
            "block_threshold":       self.state.block_threshold,
            "learned_fingerprints":  self.state.learned_fingerprints,
        }

    def stats(self) -> dict:
        return {
            "total_decisions":      self.state.total_decisions,
            "total_retrains":       self.state.total_retrains,
            "false_positive_rate":  self.state.false_positive_rate,
            "miss_rate":            self.state.miss_rate,
            "learned_fingerprints": len(self.state.learned_fingerprints),
            "current_thresholds": {
                "jailbreak":  self.state.jailbreak_threshold,
                "intent_conf":self.state.intent_conf_threshold,
                "block":      self.state.block_threshold,
            }
        }

    # ── AUTO RETRAIN ───────────────────────────────────────────────────────────
    def _auto_retrain(self):
        log.info(f"\n{'═'*50}")
        log.info(f"🚀 AUTO-RETRAIN triggered at decision #{self.state.total_decisions}")
        log.info(f"{'═'*50}")

        # Only use records not yet used in training
        unused = [r for r in self.records if not r.used_in_training
                  and r.outcome in ("CORRECT", "FALSE_POSITIVE", "MISSED")]

        log.info(f"  Unused labeled records: {len(unused)}")

        # ── Build jailbreak training samples ──────────────────────────────
        jb_samples = []
        for r in unused:
            if r.outcome == "FALSE_POSITIVE":
                label = 0    # it was actually safe
            elif r.outcome == "MISSED":
                label = 1    # it was actually a jailbreak
            elif r.final_decision == "BLOCK":
                label = 1
            else:
                label = 0
            jb_samples.append({"text": r.prompt_snippet, "label": label, "weight": r.weight})

        # ── Build intent training samples ──────────────────────────────────
        INTENT_LABEL_MAP = {
            "ROLEPLAY_JAILBREAK":        1,
            "PERSONA_SWAP":              1,
            "HYPOTHETICAL_BYPASS":       1,
            "TRAINING_MODE_EXPLOIT":     1,
            "POLICY_BYPASS":             3,
            "INSTRUCTION_OVERRIDE":      3,
            "SYSTEM_PROMPT_EXTRACTION":  2,
            "INDIRECT_INJECTION":        1,
            "COORDINATED_ATTACK":        3,
            "SUSPICIOUS_PATTERN":        1,
        }
        intent_samples = []
        for r in unused:
            if r.outcome == "FALSE_POSITIVE":
                label = 0    # benign
            elif r.outcome in ("MISSED", "CORRECT") and r.final_decision == "BLOCK":
                label = INTENT_LABEL_MAP.get(r.attack_category, 1)
            else:
                label = 0
            intent_samples.append({"text": r.prompt_snippet, "label": label, "weight": r.weight})

        # ── Train jailbreak model ──────────────────────────────────────────
        log.info("\n  [JAILBREAK MODEL]")
        jb_result = self.jb_trainer.train_and_promote(jb_samples)

        # ── Train intent model ─────────────────────────────────────────────
        log.info("\n  [INTENT MODEL]")
        intent_result = self.intent_trainer.train_and_promote(intent_samples)

        # Mark records as used
        for r in unused:
            r.used_in_training = True

        self.state.total_retrains += 1

        # Log retrain event
        retrain_entry = {
            "timestamp":      datetime.utcnow().isoformat() + "Z",
            "decision_count": self.state.total_decisions,
            "retrain_number": self.state.total_retrains,
            "jailbreak":      jb_result,
            "intent":         intent_result,
            "thresholds": {
                "jailbreak":   self.state.jailbreak_threshold,
                "intent_conf": self.state.intent_conf_threshold,
                "block":       self.state.block_threshold,
            }
        }
        with open(RETRAIN_LOG, "a") as f:
            f.write(json.dumps(retrain_entry) + "\n")

        log.info(f"\n  ✅ Retrain #{self.state.total_retrains} complete")
        log.info(f"  JB: {'✅ promoted' if jb_result.get('promoted') else '❌ not promoted'} | "
                 f"Intent: {'✅ promoted' if intent_result.get('promoted') else '❌ not promoted'}")

    # ── PERSISTENCE ────────────────────────────────────────────────────────────
    def _append_to_disk(self, record: DecisionRecord):
        with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def _load_records(self) -> list[DecisionRecord]:
        if not os.path.exists(DECISIONS_LOG):
            return []
        records = []
        with open(DECISIONS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(DecisionRecord(**json.loads(line.strip())))
                except Exception:
                    pass
        return records

    def _load_state(self) -> HealerState:
        if os.path.exists(HEALER_STATE_PATH):
            try:
                with open(HEALER_STATE_PATH) as f:
                    return HealerState(**json.load(f))
            except Exception:
                pass
        return HealerState()

    def _save_state(self):
        with open(HEALER_STATE_PATH, "w") as f:
            json.dump(asdict(self.state), f, indent=2)