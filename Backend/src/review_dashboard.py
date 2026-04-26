import os
import json
import argparse
from datetime import datetime
from pathlib import Path

_HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(_HERE)
MEMORY_LOG = os.path.join(ROOT, "logs", "agent_memory.jsonl")


def load_memory() -> list:
    if not os.path.exists(MEMORY_LOG):
        print("No memory log found yet. Run the pipeline first.")
        return []
    entries = []
    with open(MEMORY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                pass
    return entries


def save_memory(entries: list):
    with open(MEMORY_LOG, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def review_pending():
    entries  = load_memory()
    pending  = [e for e in entries if not e.get("outcome")]

    if not pending:
        print(f"\n✅ No pending reviews. {len(entries)} total cases in memory.\n")
        return

    print(f"\n{'═'*60}")
    print(f"  CYBERWATCH REVIEW DASHBOARD")
    print(f"  {len(pending)} cases pending review")
    print(f"{'═'*60}\n")

    reviewed = 0
    for i, entry in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] ─────────────────────────────────────────")
        print(f"  Prompt:    \"{entry.get('prompt_snippet', '')[:100]}\"")
        print(f"  Decision:  {entry.get('decision', '?')}  (risk: {entry.get('risk_score', '?')})")
        print(f"  Category:  {entry.get('attack_category', '?')}")
        print(f"  Confidence:{entry.get('confidence', 0):.0%}")
        print(f"  Reasoning: {entry.get('llm_reasoning', '')[:120]}")
        print(f"  Time:      {entry.get('timestamp', '?')[:19]}")
        print()
        print("  Was this decision correct?")
        print("  [1] CORRECT       — pipeline made the right call")
        print("  [2] FALSE_POSITIVE— pipeline blocked something safe")
        print("  [3] MISSED        — pipeline allowed an actual attack")
        print("  [s] Skip          — not sure, review later")
        print("  [q] Quit          — save and exit")
        print()

        while True:
            choice = input("  Your choice: ").strip().lower()
            if choice == "1":
                entry["outcome"] = "CORRECT"
                entry["weight"]  = min(2.0, entry.get("weight", 1.0) * 1.1)
                reviewed += 1
                break
            elif choice == "2":
                entry["outcome"] = "FALSE_POSITIVE"
                entry["weight"]  = max(0.1, entry.get("weight", 1.0) * 0.5)
                reviewed += 1
                break
            elif choice == "3":
                entry["outcome"] = "MISSED"
                entry["weight"]  = min(2.0, entry.get("weight", 1.0) * 1.3)
                reviewed += 1
                break
            elif choice == "s":
                print("  Skipped.\n")
                break
            elif choice == "q":
                save_memory(entries)
                print(f"\n  ✅ Saved {reviewed} reviews. Exiting.\n")
                return
            else:
                print("  Please enter 1, 2, 3, s, or q")

        print()

    save_memory(entries)
    print(f"{'═'*60}")
    print(f"  ✅ Review session complete. {reviewed} cases reviewed.")
    print(f"  Run retrain_pipeline.py to train on these reviews.")
    print(f"{'═'*60}\n")


def show_stats():
    entries = load_memory()
    if not entries:
        print("No memory entries found.")
        return

    total     = len(entries)
    reviewed  = [e for e in entries if e.get("outcome")]
    pending   = total - len(reviewed)
    correct   = sum(1 for e in reviewed if e["outcome"] == "CORRECT")
    fp        = sum(1 for e in reviewed if e["outcome"] == "FALSE_POSITIVE")
    missed    = sum(1 for e in reviewed if e["outcome"] == "MISSED")
    blocks    = sum(1 for e in entries if e.get("decision") == "BLOCK")
    allows    = sum(1 for e in entries if e.get("decision") == "ALLOW")

    cats = {}
    for e in entries:
        c = e.get("attack_category", "UNKNOWN")
        cats[c] = cats.get(c, 0) + 1

    print(f"\n{'═'*60}")
    print(f"  CYBERWATCH MEMORY STATISTICS")
    print(f"{'═'*60}")
    print(f"  Total cases:     {total}")
    print(f"  Pending review:  {pending}")
    print(f"  Reviewed:        {len(reviewed)}")
    print(f"    ✅ Correct:      {correct}")
    print(f"    ⚠ False pos:    {fp}")
    print(f"    🚨 Missed:       {missed}")
    print(f"  Blocks:          {blocks}")
    print(f"  Allows:          {allows}")
    if reviewed:
        accuracy = correct / len(reviewed) * 100
        fp_rate  = fp / len(reviewed) * 100
        print(f"  Accuracy:        {accuracy:.1f}%")
        print(f"  False pos rate:  {fp_rate:.1f}%")
    print(f"\n  Top attack categories:")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True)[:6]:
        bar = "█" * min(20, count)
        print(f"    {cat:<30} {bar} {count}")
    print(f"{'═'*60}\n")


def show_pending_count():
    entries = load_memory()
    pending = [e for e in entries if not e.get("outcome")]
    print(f"\n  📋 {len(pending)} cases pending review out of {len(entries)} total.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberWatch Human Review Dashboard")
    parser.add_argument("--stats",   action="store_true", help="Show review statistics")
    parser.add_argument("--pending", action="store_true", help="Show pending count only")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.pending:
        show_pending_count()
    else:
        review_pending()
