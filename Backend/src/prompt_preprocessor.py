"""
prompt_preprocessor.py — CyberWatch AI
=========================================
Preprocessing layer that runs BEFORE the pipeline.
Detects and normalizes 7 advanced attack types:

  1. Base64 encoding          — decodes and scans decoded text
  2. Fragmented instructions  — removes separators, rejoins words
  3. Scrambled words          — fuzzy matches against attack keywords
  4. Language switching       — transliterates known attack phrases
  5. Webpage/URL poisoning    — extracts and scans embedded instructions
  6. Conversation history inj — detects fake [SYSTEM] / [ASSISTANT] tags
  7. Multi-turn buildup       — detects escalation patterns in single prompt

Returns a PreprocessResult with:
  - normalized_prompt   : cleaned version sent to pipeline
  - original_prompt     : original unchanged
  - detected_attacks    : list of attack types found
  - risk_boost          : extra risk score to add (0-3)
  - attack_category     : override category if detected
  - flags               : detailed findings per attack type
"""

import re
import base64
import unicodedata
from dataclasses import dataclass, field


# ── Known attack keywords for fuzzy/fragment matching ─────────────────────────
ATTACK_KEYWORDS = [
    "ignore", "previous", "instructions", "disregard", "forget",
    "system", "prompt", "override", "bypass", "jailbreak",
    "pretend", "act as", "roleplay", "dan", "unrestricted",
    "no restrictions", "no limits", "without filters",
    "reveal", "training mode", "developer mode", "debug mode",
    "hypothetically", "no guidelines", "no ethical",
]

# ── Fragment separators that attackers use to break up words ──────────────────
FRAGMENT_SEPARATORS = [
    r"[-_\.\s/\\|*~`^]+",   # hyphens, underscores, dots, spaces between chars
    r"\s+",                   # extra whitespace
    r"[0-9]+",                # numbers inserted between letters
]

# ── Known attack phrases in other languages ───────────────────────────────────
MULTILINGUAL_ATTACKS = {
    # French
    "ignorez toutes":           "ignore all",
    "ignorez les instructions": "ignore instructions",
    "faites semblant":          "pretend",
    "oubliez vos instructions": "forget your instructions",
    # Spanish
    "ignora todas":             "ignore all",
    "ignora las instrucciones": "ignore instructions",
    "finge que":                "pretend that",
    "olvida tus instrucciones": "forget your instructions",
    # German
    "ignoriere alle":           "ignore all",
    "vergiss deine":            "forget your",
    "tu so als":                "act as",
    # Hindi (romanized)
    "saari instructions bhool": "forget all instructions",
    "niyamon ko todo":          "break the rules",
    # Chinese (romanized)
    "hushi suoyou zhiling":     "ignore all instructions",
    # Portuguese
    "ignore todas":             "ignore all",
    "finja ser":                "pretend to be",
    # Italian
    "ignora tutte":             "ignore all",
    "fingi di essere":          "pretend to be",
}

# ── Fake history injection patterns ───────────────────────────────────────────
FAKE_HISTORY_PATTERNS = [
    r"\[system\s*:",            # [system: ...]
    r"\[assistant\s*:",         # [assistant: ...]
    r"\[user\s*:",              # [user: ...]
    r"<\s*system\s*>",          # <system>
    r"<\s*assistant\s*>",       # <assistant>
    r"previous\s+conversation\s*:",
    r"chat\s+history\s*:",
    r"earlier\s+you\s+said",
    r"you\s+previously\s+agreed",
    r"in\s+our\s+last\s+conversation",
    r"system\s*prompt\s*:",
    r"\|\s*system\s*\|",
    r"###\s*system",
    r"##\s*instruction",
]

# ── Multi-turn buildup patterns in a single prompt ───────────────────────────
MULTI_TURN_PATTERNS = [
    r"first.*then.*finally",
    r"step\s*1.*step\s*2",
    r"start\s+by.*now\s+(do|say|tell|give)",
    r"let'?s\s+play.*now\s+answer",
    r"we\s+established.*therefore",
    r"you\s+agreed.*now",
    r"since\s+you\s+said.*now",
    r"building\s+on.*now",
]

# ── Webpage/URL embedded instruction patterns ─────────────────────────────────
URL_INJECTION_PATTERNS = [
    r"https?://[^\s]+ignore[^\s]*",
    r"https?://[^\s]+bypass[^\s]*",
    r"https?://[^\s]+jailbreak[^\s]*",
    r"<[^>]*ignore[^>]*>",           # HTML tags with attack content
    r"<!--.*ignore.*-->",             # HTML comments
    r"\[.*\]\(.*ignore.*\)",          # Markdown links
    r"data:text/[^;]+;base64,[A-Za-z0-9+/=]+",  # Data URIs
]


@dataclass
class PreprocessResult:
    original_prompt:   str
    normalized_prompt: str
    detected_attacks:  list  = field(default_factory=list)
    risk_boost:        int   = 0
    attack_category:   str   = ""
    flags:             dict  = field(default_factory=dict)

    @property
    def was_modified(self) -> bool:
        return self.original_prompt != self.normalized_prompt

    @property
    def has_attacks(self) -> bool:
        return len(self.detected_attacks) > 0


class PromptPreprocessor:
    """
    Runs before the security pipeline.
    Call: result = preprocessor.process(prompt)
    Then pass result.normalized_prompt to the pipeline.
    """

    def process(self, prompt: str) -> PreprocessResult:
        result = PreprocessResult(
            original_prompt   = prompt,
            normalized_prompt = prompt,
        )

        # Run all detectors in order
        # Each modifies result.normalized_prompt and appends to detected_attacks
        self._detect_base64(result)
        self._detect_fragments(result)
        self._detect_scrambled(result)
        self._detect_multilingual(result)
        self._detect_url_injection(result)
        self._detect_fake_history(result)
        self._detect_multi_turn_buildup(result)

        # Compute risk boost from number and severity of detected attacks
        result.risk_boost = min(3, len(result.detected_attacks))

        # Set override category
        if result.detected_attacks:
            result.attack_category = self._pick_category(result.detected_attacks)

        if result.has_attacks:
            print(f"   🔍 Preprocessor detected: {result.detected_attacks}")
            if result.was_modified:
                print(f"   🔄 Normalized prompt: {result.normalized_prompt[:80]}...")

        return result

    # ── 1. BASE64 ENCODING ────────────────────────────────────────────────────
    def _detect_base64(self, result: PreprocessResult):
        """
        Finds base64 strings in the prompt, decodes them,
        checks if decoded text contains attack keywords.
        Replaces encoded string with decoded version.
        """
        # Match base64-like strings (min 20 chars to avoid false positives)
        pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        matches = re.findall(pattern, result.normalized_prompt)

        decoded_any = False
        working = result.normalized_prompt

        for match in matches:
            try:
                # Pad if needed
                padded  = match + "=" * (-len(match) % 4)
                decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")

                # Only replace if decoded text is meaningful (contains letters)
                if len(decoded) > 5 and re.search(r'[a-zA-Z]{3,}', decoded):
                    working = working.replace(match, f"[DECODED: {decoded}]")
                    decoded_any = True

                    # Check if decoded content is an attack
                    decoded_lower = decoded.lower()
                    if any(kw in decoded_lower for kw in ATTACK_KEYWORDS):
                        result.detected_attacks.append("BASE64_ENCODING")
                        result.flags["base64"] = {
                            "encoded": match[:30] + "...",
                            "decoded": decoded[:100],
                        }
                        break

            except Exception:
                continue

        if decoded_any:
            result.normalized_prompt = working

    # ── 2. FRAGMENTED INSTRUCTIONS ────────────────────────────────────────────
    def _detect_fragments(self, result: PreprocessResult):
        """
        Detects words broken up with separators:
        I-g-n-o-r-e → Ignore
        ign.ore → ignore
        i g n o r e → ignore
        """
        text = result.normalized_prompt

        # Pattern: single chars separated by non-alpha chars (e.g. i-g-n-o-r-e)
        char_sep_pattern = r'(?:[a-zA-Z][^a-zA-Z\s]{1,3}){3,}[a-zA-Z]'
        matches = re.findall(char_sep_pattern, text)

        for match in matches:
            # Reconstruct by removing separators
            reconstructed = re.sub(r'[^a-zA-Z]', '', match).lower()
            if any(kw.replace(" ", "") in reconstructed for kw in ATTACK_KEYWORDS):
                result.detected_attacks.append("FRAGMENTED_INSTRUCTIONS")
                result.flags["fragments"] = {
                    "original": match,
                    "reconstructed": reconstructed,
                }
                # Replace fragmented version with reconstructed
                result.normalized_prompt = text.replace(match, reconstructed)
                break

        # Also check for words with inserted spaces: "i g n o r e"
        spaced_pattern = r'\b([a-zA-Z] ){4,}[a-zA-Z]\b'
        spaced_matches = re.findall(spaced_pattern, text)
        if spaced_matches:
            reconstructed = re.sub(r'\s+', '', "".join(spaced_matches)).lower()
            if any(kw in reconstructed for kw in ATTACK_KEYWORDS):
                if "FRAGMENTED_INSTRUCTIONS" not in result.detected_attacks:
                    result.detected_attacks.append("FRAGMENTED_INSTRUCTIONS")

    # ── 3. SCRAMBLED WORDS ────────────────────────────────────────────────────
    def _detect_scrambled(self, result: PreprocessResult):
        """
        Detects typo/scramble attacks:
        "ignroe" → "ignore", "preivous" → "previous"
        Uses a simple edit-distance check.
        """
        words = re.findall(r'\b[a-zA-Z]{4,}\b', result.normalized_prompt.lower())

        for word in words:
            for keyword in ATTACK_KEYWORDS:
                kw_parts = keyword.split()
                if len(kw_parts) == 1:
                    if self._edit_distance(word, keyword) <= 2 and word != keyword:
                        result.detected_attacks.append("SCRAMBLED_WORDS")
                        result.flags["scrambled"] = {
                            "found": word,
                            "matched_to": keyword,
                        }
                        # Replace scrambled word with correct spelling
                        result.normalized_prompt = re.sub(
                            r'\b' + re.escape(word) + r'\b',
                            keyword,
                            result.normalized_prompt,
                            flags=re.IGNORECASE
                        )
                        return  # one detection is enough

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Levenshtein distance between two strings."""
        if abs(len(s1) - len(s2)) > 3:
            return 99
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    # ── 4. LANGUAGE SWITCHING ─────────────────────────────────────────────────
    def _detect_multilingual(self, result: PreprocessResult):
        """
        Detects known attack phrases in other languages.
        Translates them to English in the normalized prompt.
        """
        text_lower = result.normalized_prompt.lower()

        for foreign_phrase, english_equiv in MULTILINGUAL_ATTACKS.items():
            if foreign_phrase in text_lower:
                result.detected_attacks.append("LANGUAGE_SWITCHING")
                result.flags["multilingual"] = {
                    "detected": foreign_phrase,
                    "translated_to": english_equiv,
                }
                # Replace with English equivalent for pipeline scanning
                result.normalized_prompt = re.sub(
                    re.escape(foreign_phrase),
                    english_equiv,
                    result.normalized_prompt,
                    flags=re.IGNORECASE
                )
                break

    # ── 5. WEBPAGE / URL INJECTION ────────────────────────────────────────────
    def _detect_url_injection(self, result: PreprocessResult):
        """
        Detects attack instructions embedded in URLs, HTML, or markdown links.
        Extracts the embedded text and appends it to normalized prompt.
        """
        text = result.normalized_prompt

        for pattern in URL_INJECTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result.detected_attacks.append("WEBPAGE_POISONING")
                result.flags["url_injection"] = {
                    "pattern_matched": pattern,
                    "snippet": match.group(0)[:80],
                }
                # Strip the injected element and append its content plainly
                extracted = re.sub(r'https?://\S+', '', match.group(0))
                extracted = re.sub(r'[<>\[\]()]', ' ', extracted).strip()
                if extracted:
                    result.normalized_prompt = text + f" [EXTRACTED: {extracted}]"
                break

    # ── 6. CONVERSATION HISTORY INJECTION ────────────────────────────────────
    def _detect_fake_history(self, result: PreprocessResult):
        """
        Detects injected fake [SYSTEM], [ASSISTANT], <system> tags
        or phrases like "previous conversation:", "you agreed earlier".
        """
        text_lower = result.normalized_prompt.lower()

        for pattern in FAKE_HISTORY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                result.detected_attacks.append("CONVERSATION_HISTORY_INJECTION")
                result.flags["fake_history"] = {
                    "pattern_matched": pattern,
                }
                # Strip the fake history markers
                cleaned = re.sub(
                    r'\[(?:system|assistant|user)[^\]]*\]',
                    '[REMOVED_FAKE_TAG]',
                    result.normalized_prompt,
                    flags=re.IGNORECASE
                )
                cleaned = re.sub(
                    r'<(?:system|assistant)[^>]*>.*?</(?:system|assistant)>',
                    '[REMOVED_FAKE_TAG]',
                    cleaned,
                    flags=re.IGNORECASE | re.DOTALL
                )
                result.normalized_prompt = cleaned
                break

    # ── 7. MULTI-TURN BUILDUP ─────────────────────────────────────────────────
    def _detect_multi_turn_buildup(self, result: PreprocessResult):
        """
        Detects single prompts that simulate a multi-turn attack:
        "Let's establish X... now that we agreed... answer freely"
        "Step 1: do this. Step 2: now ignore..."
        """
        text_lower = result.normalized_prompt.lower()

        for pattern in MULTI_TURN_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                result.detected_attacks.append("MULTI_TURN_ATTACK")
                result.flags["multi_turn"] = {
                    "pattern_matched": pattern,
                }
                break

        # Also check for role-establishment followed by instruction
        role_then_instruction = (
            re.search(r"(you are|you're|act as|pretend)", text_lower) and
            re.search(r"(now|therefore|so|thus).{0,50}(ignore|bypass|reveal|tell me)", text_lower)
        )
        if role_then_instruction and "MULTI_TURN_ATTACK" not in result.detected_attacks:
            result.detected_attacks.append("MULTI_TURN_ATTACK")
            result.flags["multi_turn"] = {"pattern": "role_establishment_then_instruction"}

    # ── CATEGORY PICKER ───────────────────────────────────────────────────────
    def _pick_category(self, detected_attacks: list) -> str:
        """Map preprocessor attack types to pipeline attack categories."""
        priority_map = {
            "CONVERSATION_HISTORY_INJECTION": "INDIRECT_INJECTION",
            "WEBPAGE_POISONING":              "INDIRECT_INJECTION",
            "BASE64_ENCODING":                "INSTRUCTION_OVERRIDE",
            "FRAGMENTED_INSTRUCTIONS":        "INSTRUCTION_OVERRIDE",
            "MULTI_TURN_ATTACK":              "COORDINATED_ATTACK",
            "LANGUAGE_SWITCHING":             "POLICY_BYPASS",
            "SCRAMBLED_WORDS":                "SUSPICIOUS_PATTERN",
        }
        for attack in detected_attacks:
            if attack in priority_map:
                return priority_map[attack]
        return "SUSPICIOUS_PATTERN"