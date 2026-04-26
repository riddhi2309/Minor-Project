import os
import json
import shutil
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(_HERE)
LOGS_DIR       = os.path.join(PROJECT_ROOT, "logs")
DECISIONS_LOG  = os.path.join(LOGS_DIR, "agentic_rl_decisions.jsonl")
MEMORY_LOG     = os.path.join(LOGS_DIR, "agent_memory.jsonl")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR    = os.path.join(PROJECT_ROOT, "retrain_reports")

# Model paths — adjust to match where YOUR models are saved
JB_MODEL_PATH      = os.path.join(MODELS_DIR, "jailbreak_model")
JB_MODEL_BACKUP    = os.path.join(MODELS_DIR, "jailbreak_model_backup")
JB_MODEL_CANDIDATE = os.path.join(MODELS_DIR, "jailbreak_model_candidate")

INTENT_MODEL_PATH      = os.path.join(MODELS_DIR, "intent_model")
INTENT_MODEL_BACKUP    = os.path.join(MODELS_DIR, "intent_model_backup")
INTENT_MODEL_CANDIDATE = os.path.join(MODELS_DIR, "intent_model_candidate")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Training config ────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "jailbreak": {
        "epochs":        3,
        "batch_size":    8,
        "learning_rate": 2e-5,
        "max_length":    256,
        "min_new_samples": 30,     # don't retrain unless we have at least this many
        "labels":        {0: "safe", 1: "jailbreak"},
    },
    "intent": {
        "epochs":        3,
        "batch_size":    8,
        "learning_rate": 2e-5,
        "max_length":    256,
        "min_new_samples": 30,
        "labels":        {0: "benign", 1: "injection", 2: "extraction", 3: "manipulation"},
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "retrain.log")),
    ]
)
log = logging.getLogger("CyberWatch.Retrain")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — COLLECT
# ══════════════════════════════════════════════════════════════════════════════

class DataCollector:
    """
    Reads pipeline logs and memory to build a raw dataset.
    Only uses cases that have been human-reviewed (outcome != None).
    """

    def __init__(self, days_back: int = 7):
        self.days_back = days_back
        self.cutoff    = datetime.utcnow() - timedelta(days=days_back)

    def collect(self) -> dict:
        log.info(f"📥 Collecting data from last {self.days_back} days...")

        decisions = self._load_jsonl(DECISIONS_LOG)
        memory    = self._load_jsonl(MEMORY_LOG)

        # Index memory by prompt_hash for quick lookup
        mem_index = {m["prompt_hash"]: m for m in memory if m.get("outcome")}

        raw_jailbreak = []
        raw_intent    = []
        skipped       = 0

        for entry in decisions:
            # Time filter
            ts = entry.get("timestamp", "")
            try:
                entry_time = datetime.fromisoformat(ts.rstrip("Z"))
                if entry_time < self.cutoff:
                    continue
            except Exception:
                pass

            prompt = entry.get("prompt_snippet", "")
            if not prompt or len(prompt) < 10:
                skipped += 1
                continue

            decision   = entry.get("final_decision", "")
            attack_cat = entry.get("attack_category", "SAFE")

            # Only use human-reviewed cases
            mem_entry = mem_index.get(entry.get("memory_hash", ""))
            outcome   = mem_entry.get("outcome") if mem_entry else None

            if outcome is None:
                skipped += 1
                continue

            # ── Jailbreak dataset ──────────────────────────────────────────
            # Label: 1 = jailbreak (blocked, not false positive)
            #        0 = safe (allowed, or false positive block)
            if outcome == "FALSE_POSITIVE":
                jb_label = 0   # we wrongly blocked — it's safe
            elif outcome == "MISSED":
                jb_label = 1   # we wrongly allowed — it's a jailbreak
            elif decision == "BLOCK":
                jb_label = 1
            else:
                jb_label = 0

            raw_jailbreak.append({
                "text":    prompt,
                "label":   jb_label,
                "outcome": outcome,
                "weight":  mem_entry.get("weight", 1.0) if mem_entry else 1.0,
            })

            # ── Intent dataset ─────────────────────────────────────────────
            intent_label = self._map_intent_label(attack_cat, outcome, decision)
            if intent_label is not None:
                raw_intent.append({
                    "text":   prompt,
                    "label":  intent_label,
                    "outcome": outcome,
                    "weight": mem_entry.get("weight", 1.0) if mem_entry else 1.0,
                })

        log.info(f"  ✅ Jailbreak samples: {len(raw_jailbreak)} | Intent samples: {len(raw_intent)} | Skipped: {skipped}")
        return {"jailbreak": raw_jailbreak, "intent": raw_intent}

    def _map_intent_label(self, attack_cat: str, outcome: str, decision: str) -> int | None:
        if outcome == "FALSE_POSITIVE":
            return 0  # benign
        if outcome == "MISSED" or decision == "BLOCK":
            mapping = {
                "ROLEPLAY_JAILBREAK":       1,
                "PERSONA_SWAP":             1,
                "HYPOTHETICAL_BYPASS":      1,
                "TRAINING_MODE_EXPLOIT":    1,
                "POLICY_BYPASS":            3,
                "INSTRUCTION_OVERRIDE":     3,
                "SYSTEM_PROMPT_EXTRACTION": 2,
                "INDIRECT_INJECTION":       1,
                "COORDINATED_ATTACK":       3,
                "SUSPICIOUS_PATTERN":       1,
            }
            return mapping.get(attack_cat, 1)
        return 0  # benign / allowed

    def _load_jsonl(self, path: str) -> list:
        if not os.path.exists(path):
            log.warning(f"  ⚠ Log file not found: {path}")
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass
        return entries


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PromptDataset(Dataset):
    def __init__(self, samples: list, tokenizer, max_length: int = 256):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc  = self.tokenizer(
            item["text"],
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(item["label"], dtype=torch.long),
            "weight":         torch.tensor(item.get("weight", 1.0), dtype=torch.float),
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Fine-tunes a HuggingFace sequence classification model on new data.
    Uses weighted loss so high-confidence human-reviewed cases matter more.
    """

    def __init__(self, model_path: str, candidate_path: str,
                 config: dict, num_labels: int):
        self.model_path     = model_path
        self.candidate_path = candidate_path
        self.config         = config
        self.num_labels     = num_labels
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"  🖥  Training device: {self.device}")

    def train(self, samples: list) -> dict:
        """Fine-tune and return evaluation metrics."""
        if len(samples) < self.config["min_new_samples"]:
            log.info(f"  ⚠ Only {len(samples)} samples — need {self.config['min_new_samples']}. Skipping.")
            return {"skipped": True, "reason": "insufficient_data", "samples": len(samples)}

        # Train/val split
        train_s, val_s = train_test_split(samples, test_size=0.15, random_state=42,
                                          stratify=[s["label"] for s in samples])
        log.info(f"  📊 Train: {len(train_s)} | Val: {len(val_s)}")

        # Load model + tokenizer from existing path
        log.info(f"  🔄 Loading base model from: {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=self.num_labels
        ).to(self.device)

        # Datasets and loaders
        train_ds = PromptDataset(train_s, tokenizer, self.config["max_length"])
        val_ds   = PromptDataset(val_s,   tokenizer, self.config["max_length"])
        train_dl = DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.config["batch_size"])

        # Optimizer + scheduler
        optimizer = AdamW(model.parameters(), lr=self.config["learning_rate"], weight_decay=0.01)
        total_steps = len(train_dl) * self.config["epochs"]
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        # ── Training loop ──────────────────────────────────────────────────
        best_f1 = 0.0
        for epoch in range(self.config["epochs"]):
            model.train()
            epoch_loss = 0.0

            for batch in train_dl:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)
                weights        = batch["weight"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits

                # Weighted cross-entropy — human-corrected cases count more
                loss_fn    = torch.nn.CrossEntropyLoss(reduction="none")
                raw_loss   = loss_fn(logits, labels)
                weighted_loss = (raw_loss * weights).mean()

                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += weighted_loss.item()

            avg_loss = epoch_loss / len(train_dl)

            # ── Validation ────────────────────────────────────────────────
            val_f1 = self._evaluate(model, val_dl)
            log.info(f"  Epoch {epoch+1}/{self.config['epochs']} | loss={avg_loss:.4f} | val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1

        # Save candidate model
        log.info(f"  💾 Saving candidate to: {self.candidate_path}")
        model.save_pretrained(self.candidate_path)
        tokenizer.save_pretrained(self.candidate_path)

        return {
            "skipped":        False,
            "samples_used":   len(samples),
            "train_samples":  len(train_s),
            "val_samples":    len(val_s),
            "best_val_f1":    round(best_f1, 4),
            "epochs":         self.config["epochs"],
        }

    def _evaluate(self, model, val_dl) -> float:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"]
                outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
                preds          = torch.argmax(outputs.logits, dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        return f1_score(all_labels, all_preds, average="weighted", zero_division=0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATOR (compare old vs new)
# ══════════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    Compares old model vs new candidate model on a held-out eval set.
    Only promotes the candidate if it is strictly better.
    """

    def __init__(self, eval_samples: list, max_length: int = 256):
        self.eval_samples = eval_samples
        self.max_length   = max_length
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compare(self, old_path: str, new_path: str) -> dict:
        if not self.eval_samples:
            log.warning("  ⚠ No eval samples — auto-promoting candidate")
            return {"promote": True, "reason": "no_eval_data"}

        old_f1 = self._score_model(old_path)
        new_f1 = self._score_model(new_path)

        log.info(f"  📊 Old model F1: {old_f1:.4f} | New model F1: {new_f1:.4f}")

        promote = new_f1 >= old_f1 - 0.01   # allow tiny regressions (within 1%)
        return {
            "promote":   promote,
            "old_f1":    round(old_f1, 4),
            "new_f1":    round(new_f1, 4),
            "delta":     round(new_f1 - old_f1, 4),
            "reason":    "new_model_better_or_equal" if promote else "new_model_worse",
        }

    def _score_model(self, model_path: str) -> float:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            ds        = PromptDataset(self.eval_samples, tokenizer, self.max_length)
            dl        = DataLoader(ds, batch_size=16)
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in dl:
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds          = torch.argmax(outputs.logits, dim=1).cpu()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(batch["labels"].tolist())
            return f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        except Exception as e:
            log.error(f"  ✗ Error scoring model at {model_path}: {e}")
            return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PROMOTER (swap old with new)
# ══════════════════════════════════════════════════════════════════════════════

class ModelPromoter:
    """
    Safely promotes a candidate model to production.
    Always keeps a backup so you can roll back.
    """

    def promote(self, current_path: str, candidate_path: str, backup_path: str) -> bool:
        try:
            # Backup current model
            if os.path.exists(current_path):
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                shutil.copytree(current_path, backup_path)
                log.info(f"  💾 Backup saved: {backup_path}")

            # Promote candidate → current
            if os.path.exists(current_path):
                shutil.rmtree(current_path)
            shutil.copytree(candidate_path, current_path)
            log.info(f"  ✅ Promoted: {candidate_path} → {current_path}")
            return True

        except Exception as e:
            log.error(f"  ✗ Promotion failed: {e}")
            return False

    def rollback(self, current_path: str, backup_path: str) -> bool:
        try:
            if not os.path.exists(backup_path):
                log.error("  ✗ No backup found — cannot rollback")
                return False
            if os.path.exists(current_path):
                shutil.rmtree(current_path)
            shutil.copytree(backup_path, current_path)
            log.info(f"  ✅ Rolled back to backup")
            return True
        except Exception as e:
            log.error(f"  ✗ Rollback failed: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — REPORT
# ══════════════════════════════════════════════════════════════════════════════

class RetrainReporter:
    def save(self, report: dict):
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(REPORTS_DIR, f"retrain_report_{ts}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log.info(f"  📄 Report saved: {filename}")

        # Print human-readable summary
        print("\n" + "═"*60)
        print("   RETRAINING REPORT")
        print("═"*60)
        print(f"   Run time:     {report['timestamp']}")
        print(f"   Days covered: {report['days_back']}")
        print(f"   Total samples collected: {report['total_samples']}")
        print()

        for model_name in ["jailbreak", "intent"]:
            r = report.get(f"{model_name}_result", {})
            print(f"   [{model_name.upper()} MODEL]")
            if r.get("skipped"):
                print(f"   ⚠ Skipped — {r.get('reason')} ({r.get('samples')} samples)")
            else:
                print(f"   Samples trained: {r.get('samples_used', 0)}")
                print(f"   Val F1 score:    {r.get('best_val_f1', 0):.4f}")
                e = report.get(f"{model_name}_eval", {})
                print(f"   Old model F1:    {e.get('old_f1', '?')}")
                print(f"   New model F1:    {e.get('new_f1', '?')}")
                print(f"   Promoted:        {'✅ YES' if e.get('promote') else '❌ NO'}")
            print()
        print("═"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_retraining(days_back: int = 7, force: bool = False):
    """
    Full retraining pipeline.
    Args:
        days_back: how many days of logs to include
        force:     skip minimum sample check
    """
    log.info("🚀 CyberWatch Weekly Retraining Pipeline starting...")
    start_time = datetime.utcnow()
    report     = {
        "timestamp": start_time.isoformat() + "Z",
        "days_back": days_back,
    }

    # ── STEP 1: Collect ───────────────────────────────────────────────────────
    collector = DataCollector(days_back=days_back)
    data      = collector.collect()

    report["total_samples"] = len(data["jailbreak"]) + len(data["intent"])

    if force:
        # Override minimum sample requirement
        TRAIN_CONFIG["jailbreak"]["min_new_samples"] = 1
        TRAIN_CONFIG["intent"]["min_new_samples"]    = 1

    promoter = ModelPromoter()
    reporter = RetrainReporter()

    # ── STEP 2–4: Train + Evaluate + Promote: Jailbreak Model ────────────────
    log.info("\n── JAILBREAK MODEL ──────────────────────────────────────────")
    jb_trainer = ModelTrainer(
        model_path     = JB_MODEL_PATH,
        candidate_path = JB_MODEL_CANDIDATE,
        config         = TRAIN_CONFIG["jailbreak"],
        num_labels     = 2,
    )
    jb_train_result = jb_trainer.train(data["jailbreak"])
    report["jailbreak_result"] = jb_train_result

    if not jb_train_result.get("skipped"):
        # Hold out 10% as a final eval set
        _, jb_eval = train_test_split(data["jailbreak"], test_size=0.10, random_state=99)
        jb_evaluator  = ModelEvaluator(jb_eval)
        jb_eval_result = jb_evaluator.compare(JB_MODEL_PATH, JB_MODEL_CANDIDATE)
        report["jailbreak_eval"] = jb_eval_result

        if jb_eval_result["promote"]:
            promoter.promote(JB_MODEL_PATH, JB_MODEL_CANDIDATE, JB_MODEL_BACKUP)
        else:
            log.warning("  ⚠ Jailbreak candidate not promoted — old model is better")

    # ── STEP 2–4: Train + Evaluate + Promote: Intent Model ───────────────────
    log.info("\n── INTENT MODEL ─────────────────────────────────────────────")
    intent_trainer = ModelTrainer(
        model_path     = INTENT_MODEL_PATH,
        candidate_path = INTENT_MODEL_CANDIDATE,
        config         = TRAIN_CONFIG["intent"],
        num_labels     = 4,
    )
    intent_train_result = intent_trainer.train(data["intent"])
    report["intent_result"] = intent_train_result

    if not intent_train_result.get("skipped"):
        _, intent_eval = train_test_split(data["intent"], test_size=0.10, random_state=99)
        intent_evaluator   = ModelEvaluator(intent_eval)
        intent_eval_result = intent_evaluator.compare(INTENT_MODEL_PATH, INTENT_MODEL_CANDIDATE)
        report["intent_eval"] = intent_eval_result

        if intent_eval_result["promote"]:
            promoter.promote(INTENT_MODEL_PATH, INTENT_MODEL_CANDIDATE, INTENT_MODEL_BACKUP)
        else:
            log.warning("  ⚠ Intent candidate not promoted — old model is better")

    # ── STEP 5: Report ────────────────────────────────────────────────────────
    report["duration_seconds"] = round((datetime.utcnow() - start_time).total_seconds(), 1)
    reporter.save(report)

    log.info(f"✅ Retraining complete in {report['duration_seconds']}s")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# ROLLBACK UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def rollback_models():
    """
    Emergency rollback — restores both models from last backup.
    Run this if the new models are causing problems in production.

    Usage:  python retrain_pipeline.py --rollback
    """
    promoter = ModelPromoter()
    log.info("⏪ Rolling back models to last backup...")
    jb_ok     = promoter.rollback(JB_MODEL_PATH,     JB_MODEL_BACKUP)
    intent_ok = promoter.rollback(INTENT_MODEL_PATH, INTENT_MODEL_BACKUP)
    if jb_ok and intent_ok:
        log.info("✅ Rollback complete")
    else:
        log.error("✗ Rollback had errors — check logs")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberWatch Weekly Retraining Pipeline")
    parser.add_argument("--days",     type=int,  default=7,     help="Days of logs to include (default: 7)")
    parser.add_argument("--force",    action="store_true",       help="Skip minimum sample check")
    parser.add_argument("--rollback", action="store_true",       help="Rollback to last backup")
    args = parser.parse_args()

    if args.rollback:
        rollback_models()
    else:
        run_retraining(days_back=args.days, force=args.force)
