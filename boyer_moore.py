# =============================================================================
# boyer_moore_toxic_detector.py
# Enhanced Hybrid Toxic Language Detection
# =============================================================================
# Combines fast Boyer–Moore pattern matching with:
#  • text deobfuscation
#  • tiered severity scoring
#  • context awareness for reduced false positives
#  • overall toxicity scoring
# =============================================================================

import re
from typing import List, Dict

# =============================================================================
# OBFUSCATION HANDLING
# =============================================================================

CHAR_SUBSTITUTIONS = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '8': 'b', '@': 'a', '$': 's', '!': 'i',
    '|': 'i', '+': 't', 'ph': 'f', 'x': 'ks', 'q': 'k',
}

SEPARATOR_PATTERN = re.compile(r'[\s.\-_*]+')
REPEAT_PATTERN = re.compile(r'(.)\1{2,}')


def deobfuscate(text: str) -> str:
    import string
    text = text.translate(str.maketrans('', '', string.punctuation))

    """Removes separators, collapses duplicates, applies leetspeak fixes."""
    text = text.lower()
    text = SEPARATOR_PATTERN.sub(' ', text)
    text = ' '.join(text.split())
    text = REPEAT_PATTERN.sub(r'\1\1', text)
    
    for fake, real in CHAR_SUBSTITUTIONS.items():
        text = text.replace(fake, real)
    return text


# =============================================================================
# BOYER–MOORE  —  Preprocessing & Search
# =============================================================================

def build_bad_character_table(pattern: str) -> dict:
    return {char: i for i, char in enumerate(pattern)}


def build_good_suffix_table(pattern: str) -> list:
    m = len(pattern)
    shift = [m] * (m + 1)
    border = [0] * (m + 1)
    i, j = m, m + 1
    border[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == m:
                shift[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j
    j = border[0]
    for i in range(m + 1):
        if shift[i] == m:
            shift[i] = j
        if i == j:
            j = border[j]
    return shift


def boyer_moore_search(text: str, pattern: str) -> list:
    """Single‑pattern Boyer–Moore search."""
    if not text or not pattern or len(pattern) > len(text):
        return []
    n, m = len(text), len(pattern)
    bad_char = build_bad_character_table(pattern)
    good_suffix = build_good_suffix_table(pattern)

    matches, i = [], 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            matches.append(i)
            i += good_suffix[0]
        else:
            bc_shift = j - bad_char.get(text[i + j], -1)
            gs_shift = good_suffix[j + 1]
            i += max(bc_shift, gs_shift)
    return matches


# =============================================================================
# SEVERITY‑BASED DICTIONARY
# =============================================================================

KEYWORD_TIERS = {
    "critical": {
        "keywords": [
            "putang ina", "tangina", "hindot", "pakyu", "delete yourself",
            "kill yourself"
        ],
        "severity": 1.0,
    },
    "high": {
        "keywords": [
            "gago", "bobo", "tanga", "ulol", "puta", "gaga",
            "kupal", "punyeta", "retard", "cancer"
        ],
        "severity": 0.7,
    },
    "medium": {
        "keywords": [
            "leche", "hayop", "buwisit", "lintik", "tae",
            "animal", "peste", "cripple", "die"
        ],
        "severity": 0.4,
    },
}


# =============================================================================
# CONTEXT RULES  —  Filters for false positives
# =============================================================================

CONTEXT_RULES = {
    "die": {
        "toxic_if_preceded_by": ["go", "pls", "please", "just", "u", "you"],
        "safe_if_preceded_by": ["i", "will", "gonna", "dont", "don't", "didn't"],
    },
    "cancer": {
        "toxic_if_standalone": True,
        "safe_if_followed_by": ["sign", "zodiac", "awareness"],
    },
}


def is_contextually_toxic(text: str, keyword: str, position: int) -> bool:
    """Checks if context around a match makes it actually toxic."""
    if keyword not in CONTEXT_RULES:
        return True
    rules = CONTEXT_RULES[keyword]
    words = text.split()
    char_count, word_index = 0, 0
    for i, word in enumerate(words):
        if char_count + len(word) > position:
            word_index = i
            break
        char_count += len(word) + 1
    if word_index > 0:
        prev = words[word_index - 1]
        if "safe_if_preceded_by" in rules and prev in rules["safe_if_preceded_by"]:
            return False
        if "toxic_if_preceded_by" in rules and prev in rules["toxic_if_preceded_by"]:
            return True
    if word_index < len(words) - 1:
        nxt = words[word_index + 1]
        if "safe_if_followed_by" in rules and nxt in rules["safe_if_followed_by"]:
            return False
    return True


# =============================================================================
# SCANNING & SCORING
# =============================================================================

def boyer_moore_scan_with_severity(text: str) -> dict:
    """Scans text for all tiers, returns matches with severity metadata."""
    matches, max_sev, total = [], 0.0, 0.0
    for tier, data in KEYWORD_TIERS.items():
        sev = data["severity"]
        for kw in data["keywords"]:
            positions = boyer_moore_search(text, kw)
            if positions:
                matches.append({
                    "keyword": kw,
                    "positions": positions,
                    "tier": tier,
                    "severity": sev,
                })
                max_sev = max(max_sev, sev)
                total += sev * len(positions)
    return {"matches": matches, "max_severity": max_sev, "total_score": total}


def compute_toxicity_score(result: dict, text_len: int) -> float:
    """Combines severity, density, and accumulation into final score."""
    if not result["matches"]:
        return 0.0
    max_sev = result["max_severity"]
    total = result["total_score"]
    match_count = len(result["matches"])
    density = min(1.0, total / max(10, text_len / 10))
    accumulation = min(0.3, match_count * 0.1)
    return min(1.0, (max_sev * 0.6) + (density * 0.25) + accumulation)


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def boyer_moore_detect(
    text: str,
    threshold: float = 0.4,
    return_details: bool = False
) -> dict:
    """
    Full detection pipeline:
      1. Deobfuscation
      2. Boyer–Moore scan with severity
      3. Context filtering
      4. Score computation
    """
    normalized = deobfuscate(text)
    scan_result = boyer_moore_scan_with_severity(normalized)

    filtered = []
    for m in scan_result["matches"]:
        for pos in m["positions"]:
            if is_contextually_toxic(normalized, m["keyword"], pos):
                filtered.append(m)
                break

    if filtered:
        max_sev = max(m["severity"] for m in filtered)
        total = sum(m["severity"] * len(m["positions"]) for m in filtered)
    else:
        max_sev = total = 0.0

    adjusted = {"matches": filtered, "max_severity": max_sev, "total_score": total}
    score = compute_toxicity_score(adjusted, len(text))

    out = {"is_toxic": score >= threshold, "score": round(score, 3)}
    if return_details:
        out["details"] = filtered
        out["normalized_text"] = normalized
    return out


# =============================================================================
# MANUAL TESTING
# =============================================================================
if __name__ == "__main__":
    samples = [
        "ang gago mo",
        "b0b0 ka tangina mo",
        "go die you noob",
        "i will die here guys",
        "puta ka ulol gago!!!",
        "g.a.g.o b.o.b.o ka",
        "good luck team, gg"
    ]
    print("=" * 70)
    print(" ENHANCED BOYER–MOORE TOXICITY DETECTOR TESTS ".center(70, "="))
    print("=" * 70)

    for msg in samples:
        result = boyer_moore_detect(msg, return_details=True)
        print(f"\nInput: {msg}")
        print(f"Toxic: {result['is_toxic']} (score={result['score']})")
        if result["is_toxic"]:
            for d in result["details"]:
                print(f"  - {d['keyword']} (tier={d['tier']}, sev={d['severity']})")
    print("=" * 70)
