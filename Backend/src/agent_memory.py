import os
import json
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


_HERE    = os.path.dirname(os.path.abspath(__file__))
MEM_PATH = os.path.join(_HERE, "..", "logs", "agent_memory.jsonl")
os.makedirs(os.path.dirname(MEM_PATH), exist_ok=True)


@dataclass
class MemoryEntry:
    prompt_hash:     str
    prompt_snippet:  str
    decision:        str          # BLOCK / ALLOW
    attack_category: str
    confidence:      float
    risk_score:      int
    tools_used:      list
    llm_reasoning:   str          # what the LLM said
    timestamp:       str
    outcome:         Optional[str] = None   # CORRECT / FALSE_POSITIVE / MISSED
    weight:          float = 1.0            # boosted/reduced by feedback


class AgentMemory:
    """
    Episodic memory for the RL agent.

    The agent uses this memory in two ways:
      1. RECALL  — before deciding, retrieve similar past cases
      2. LEARN   — after deciding, store the new case
      3. REFLECT — when outcome feedback arrives, adjust weights

    This is not gradient-based learning — it is in-context learning.
    The LLM reads past examples and reasons more accurately as a result.
    The effect is similar to few-shot prompting, but the examples are
    real cases your pipeline has seen, not hand-crafted ones.
    """

    def __init__(self, max_entries: int = 2000):
        self.max_entries = max_entries
        self.entries: list[MemoryEntry] = []
        self._load()
        print(f"  ✅ AgentMemory loaded ({len(self.entries)} past cases)")

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, prompt: str, decision: str, attack_category: str,
              confidence: float, risk_score: int, tools_used: list,
              llm_reasoning: str) -> str:
        """Save a new decision to memory. Returns the entry's hash."""
        ph = self._hash(prompt)
        entry = MemoryEntry(
            prompt_hash     = ph,
            prompt_snippet  = prompt[:120],
            decision        = decision,
            attack_category = attack_category,
            confidence      = round(confidence, 3),
            risk_score      = risk_score,
            tools_used      = tools_used,
            llm_reasoning   = llm_reasoning[:300],
            timestamp       = datetime.utcnow().isoformat() + "Z",
        )
        self.entries.append(entry)
        # Keep memory bounded
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        self._append_to_disk(entry)
        return ph

    def recall(self, prompt: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        Find the most relevant past cases for this prompt.
        Uses keyword overlap (fast, no embeddings needed for local setup).
        Returns up to top_k entries sorted by relevance × weight.
        """
        if not self.entries:
            return []

        prompt_words = set(self._tokenize(prompt))
        scored = []
        for e in self.entries:
            entry_words = set(self._tokenize(e.prompt_snippet))
            overlap = len(prompt_words & entry_words)
            if overlap > 0:
                scored.append((overlap * e.weight, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def feedback(self, prompt_hash: str, outcome: str):
        """
        Receive outcome feedback: CORRECT, FALSE_POSITIVE, or MISSED.
        Adjusts the weight of the case so future recalls are influenced.
        - CORRECT       → weight stays or increases slightly
        - FALSE_POSITIVE→ weight drops (don't repeat this mistake)
        - MISSED        → weight stays, but the case is flagged for review
        """
        for e in self.entries:
            if e.prompt_hash == prompt_hash:
                e.outcome = outcome
                if outcome == "CORRECT":
                    e.weight = min(2.0, e.weight * 1.1)
                elif outcome == "FALSE_POSITIVE":
                    e.weight = max(0.1, e.weight * 0.5)
                self._rewrite_disk()
                return

    def stats(self) -> dict:
        """Summary of what the agent has learned so far."""
        if not self.entries:
            return {"total": 0}
        blocks   = sum(1 for e in self.entries if e.decision == "BLOCK")
        allows   = sum(1 for e in self.entries if e.decision == "ALLOW")
        cats     = {}
        for e in self.entries:
            cats[e.attack_category] = cats.get(e.attack_category, 0) + 1
        return {
            "total":            len(self.entries),
            "blocks":           blocks,
            "allows":           allows,
            "top_attack_types": sorted(cats.items(), key=lambda x: x[1], reverse=True)[:5],
            "avg_confidence":   round(sum(e.confidence for e in self.entries) / len(self.entries), 3),
        }

    def format_for_context(self, similar_cases: list[MemoryEntry]) -> str:
        """
        Format past cases into a string the LLM can read as experience.
        This is injected into the agent's system prompt.
        """
        if not similar_cases:
            return "No similar past cases found."

        lines = ["PAST SIMILAR CASES (learn from these):"]
        for i, e in enumerate(similar_cases, 1):
            lines.append(
                f"\n[Case {i}] Prompt: \"{e.prompt_snippet[:80]}...\"\n"
                f"  → Decision: {e.decision} | Category: {e.attack_category} "
                f"| Confidence: {e.confidence:.0%}\n"
                f"  → Reasoning: {e.llm_reasoning[:150]}\n"
                f"  → Outcome: {e.outcome or 'not yet verified'}"
            )
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r'\b[a-z]{3,}\b', text.lower())

    def _load(self):
        if not os.path.exists(MEM_PATH):
            return
        with open(MEM_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    self.entries.append(MemoryEntry(**d))
                except Exception:
                    pass

    def _append_to_disk(self, entry: MemoryEntry):
        with open(MEM_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def _rewrite_disk(self):
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(asdict(e)) + "\n")
