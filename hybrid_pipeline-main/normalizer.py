# =============================================================================
# normalizer.py
# Hybrid Toxic Language Detection Pipeline
# Component 1 — Leetspeak Normalizer
# =============================================================================
# Converts leetspeak-obfuscated characters into their standard plain text
# equivalents before pattern matching begins. This is a prerequisite step —
# without normalization, Boyer-Moore and Aho-Corasick cannot match obfuscated
# toxic terms since both algorithms operate on exact character sequences.
#
# Design Decisions:
#   - Case-insensitive: input is lowercased before substitution
#   - Substring-safe: normalization is character-level, preserving full string
#   - Single-character substitutions only (as defined in the study scope)
# =============================================================================
 
# -----------------------------------------------------------------------------
# LEETSPEAK SUBSTITUTION MAP
# Derived from the predefined substitution table in the study.
# Each key is an obfuscated character; each value is its plain equivalent.
# -----------------------------------------------------------------------------
LEET_MAP = {
    '0': 'o',   # zero  → o
    '4': 'a',   # four  → a
    '1': 'i',   # one   → i
    '3': 'e',   # three → e
    '@': 'a',   # at    → a
    '$': 's',   # dollar → s
    '+': 't',   # plus  → t
    '5': 's',   # five  → s
    '7': 't',   # seven → t
    '9': 'g',   # nine  → g
    '6': 'b',   # six   → b
    '8': 'b',   # eight → b
    '!': 'i',   # exclamation → i
    '°': 'o',   # degree → o
    'ß': 'b',   # eszett → b
}
 
 
# -----------------------------------------------------------------------------
# CORE FUNCTION
# -----------------------------------------------------------------------------
def normalize(message: str) -> str:
    """
    Normalizes a raw chat message by:
      1. Converting the message to lowercase
      2. Replacing each leetspeak character with its plain text equivalent
 
    Parameters:
        message (str): Raw chat message as typed by the player
 
    Returns:
        str: Normalized plain text message ready for pattern matching
    """
    if not message or not isinstance(message, str):
        return ""
 
    # Step 1 — Lowercase entire message for case-insensitive matching
    message = message.lower()
 
    # Step 2 — Character-by-character substitution using the leet map
    normalized = []
    for char in message:
        normalized.append(LEET_MAP.get(char, char))
 
    return ''.join(normalized)
 
 
# -----------------------------------------------------------------------------
# BATCH FUNCTION
# Normalizes a list of messages at once. Used by benchmark.py.
# -----------------------------------------------------------------------------
def normalize_batch(messages: list) -> list:
    """
    Normalizes a list of raw chat messages.
 
    Parameters:
        messages (list): List of raw chat message strings
 
    Returns:
        list: List of normalized message strings
    """
    return [normalize(msg) for msg in messages]
 
 
# =============================================================================
# QUICK TEST — run this file directly to verify normalizer is working
# =============================================================================
if __name__ == "__main__":
 
    test_inputs = [
        # (raw_input, expected_output)
        ("g4g0",                "gago"),
        ("b0b0",                "bobo"),
        ("74ng4",               "tanga"),
        ("1d107",               "idiot"),
        ("$7up1d",              "stupid"),
        ("ul0l",                "ulol"),
        ("74ng1n4",             "tangina"),
        ("9490",                "gago"),
        ("8080",                "bobo"),
        ("74n94",               "tanga"),
        ("ang g4g0 mo pre",     "ang gago mo pre"),
        ("pre ang b0b0 mo",     "pre ang bobo mo"),
        ("GAGO",                "gago"),       # case insensitive
        ("G4G0",                "gago"),       # leet + uppercase
        ("good game everyone",  "good game everyone"),  # clean — no change
        ("",                    ""),           # empty string
    ]
 
    print("=" * 65)
    print("NORMALIZER TEST RESULTS")
    print("=" * 65)
    print(f"{'Input':<30} {'Expected':<20} {'Result':<20} {'Pass'}")
    print("-" * 65)
 
    all_passed = True
    for raw, expected in test_inputs:
        result = normalize(raw)
        passed = result == expected
        if not passed:
            all_passed = False
        status = "✅" if passed else "❌"
        print(f"{raw:<30} {expected:<20} {result:<20} {status}")
 
    print("-" * 65)
    if all_passed:
        print("All tests passed. Normalizer is working correctly.")
    else:
        print("Some tests failed. Review the substitution map.")
    print("=" * 65)