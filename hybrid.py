# =============================================================================
# hybrid.py
# Hybrid Toxic Language Detection Pipeline
# Component 4 — Hybrid Pipeline Orchestrator (Updated for Enhanced BM)
# =============================================================================
# Integrates the Leetspeak Normalizer, Enhanced Boyer–Moore (Stage 1),
# and Aho–Corasick (Stage 2) into a unified hybrid detection pipeline.
# =============================================================================

import time
import tracemalloc
from normalizer import normalize
from boyer_moore import boyer_moore_detect
from aho_corasick import build_trie, build_failure_links, aho_corasick_scan_prebuilt

# Local fallback for Stage 1 pre‑screen
HIGH_RISK_KEYWORDS = [
    "gago", "bobo", "tanga", "ulol",
    "tangina", "putang ina", "puta",
    "leche", "kupal", "idiot", "stupid",
    "noob", "trash", "pakyu", "retard",
    "cancer", "die", "delete yourself"
]

# =============================================================================
# COLOR UTILITIES (for clearer console output)
# =============================================================================
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GRAY = "\033[90m"


# =============================================================================
# HYBRID PIPELINE CLASS
# =============================================================================
class HybridPipeline:
    """
    Orchestrates the full hybrid toxic‑language detection pipeline.
    Combines normalization → Boyer–Moore pre‑screen → Aho–Corasick stage.
    """

    def __init__(self, dictionary: list, high_risk_keywords: list | None = None):
        self.dictionary = [w.lower().strip() for w in dictionary if w.strip()]
        self.high_risk = high_risk_keywords or HIGH_RISK_KEYWORDS

        # Build Aho–Corasick trie once for reuse
        self.root = build_trie(self.dictionary)
        self.root = build_failure_links(self.root)

    # -------------------------------------------------------------------------
    def process(self, raw_message: str) -> dict:
        """
        Runs full hybrid pipeline on a single message.
        Returns a structured dictionary summarizing results.
        """
        # --- Stage 0: Normalization ------------------------------------------
        normalized = normalize(raw_message)

        # --- Stage 1: Enhanced Boyer–Moore pre‑screen ------------------------
        bm_output = boyer_moore_detect(normalized, threshold=0.4, return_details=True)

        if not bm_output["is_toxic"]:
            # No high‑risk keyword detected — early exit
            return {
                "raw_text": raw_message,
                "normalized_text": bm_output.get("normalized_text", normalized),
                "flagged": False,
                "stage_triggered": "none",
                "bm_score": bm_output["score"],
                "bm_matches": [],
                "ac_matches": [],
                "matches": [],
            }

        bm_matches = bm_output.get("details", [])
        normalized_final = bm_output.get("normalized_text", normalized)

        # --- Stage 2: Aho–Corasick full dictionary scan ----------------------
        ac_matches_raw = aho_corasick_scan_prebuilt(normalized_final, self.root)

        # --- Merge & Deduplicate --------------------------------------------
        merged = _merge_and_format(bm_matches, ac_matches_raw, normalized_final)

        return {
            "raw_text": raw_message,
            "normalized_text": normalized_final,
            "flagged": len(merged) > 0,
            "stage_triggered": "bm+ac",
            "bm_score": bm_output["score"],
            "bm_matches": bm_matches,
            "ac_matches": ac_matches_raw,
            "matches": merged,
        }

    # -------------------------------------------------------------------------
    def process_batch(self, messages: list) -> list:
        return [self.process(m) for m in messages]

    # -------------------------------------------------------------------------
    def process_timed(self, raw_message: str) -> dict:
        start = time.perf_counter()
        result = self.process(raw_message)
        end = time.perf_counter()
        result["processing_time_us"] = round((end - start) * 1_000_000, 3)
        return result


# =============================================================================
# ONE‑SHOT FUNCTION (Convenience Wrapper)
# =============================================================================
def hybrid_scan(raw_message: str, dictionary: list, high_risk_keywords: list | None = None) -> dict:
    pipeline = HybridPipeline(dictionary, high_risk_keywords)
    return pipeline.process(raw_message)


# =============================================================================
# HELPER: Deduplicate BM and AC matches
# =============================================================================
def _merge_and_format(bm_matches: list, ac_matches: list, text: str) -> list:
    """Merge, deduplicate, and reformat match results."""
    results, seen = [], set()

    # Convert BM results → flat format
    for entry in bm_matches:
        keyword = entry["keyword"]
        for pos in entry.get("positions", []):
            pair = (keyword, pos)
            if pair not in seen:
                seen.add(pair)
                results.append({
                    "pattern": keyword,
                    "start_index": pos,
                    "end_index": pos + len(keyword) - 1
                })

    # Add AC results
    for entry in ac_matches:
        keyword = entry["keyword"]
        for pos in entry["positions"]:
            pair = (keyword, pos)
            if pair not in seen:
                seen.add(pair)
                results.append({
                    "pattern": keyword,
                    "start_index": pos,
                    "end_index": pos + len(keyword) - 1
                })

    results.sort(key=lambda x: x["start_index"])
    return results


# =============================================================================
# PRINT UTILITY
# =============================================================================
def print_result(result: dict) -> None:
    """Nicely formatted console printer for result dictionaries."""
    print(Colors.GRAY + "-" * 70 + Colors.RESET)
    print(f"{Colors.BOLD}Raw Text:        {Colors.RESET}{result['raw_text']}")
    print(f"Normalized:      {Colors.CYAN}{result['normalized_text']}{Colors.RESET}")
    print(f"Flagged:         {Colors.RED if result['flagged'] else Colors.GREEN}{result['flagged']}{Colors.RESET}")
    print(f"Stage Triggered: {Colors.YELLOW}{result['stage_triggered']}{Colors.RESET}")
    print(f"BM Score:        {Colors.BLUE}{result.get('bm_score', 'N/A')}{Colors.RESET}")

    if result["matches"]:
        print(Colors.BOLD + "Matches:" + Colors.RESET)
        for m in result["matches"]:
            print(f"  → {Colors.RED}'{m['pattern']}'{Colors.RESET} "
                  f"({m['start_index']}–{m['end_index']})")
    else:
        print("Matches: None")

    if "processing_time_us" in result:
        print(f"Processing Time: {result['processing_time_us']} µs")

    print(Colors.GRAY + "-" * 70 + Colors.RESET)


# =============================================================================
# QUICK TEST — Verify integration
# =============================================================================
if __name__ == "__main__":
    SAMPLE_DICT = [
        "gago", "bobo", "tanga", "ulol", "tangina", "putang ina",
        "puta", "leche", "gaga", "kupal", "pakyu", "hayop",
        "tae", "animal", "peste", "idiot", "stupid", "noob", "trash"
    ]

    pipeline = HybridPipeline(SAMPLE_DICT)

    tests = [
        ("ang g4g0 mo pre", True),
        ("b0b0 ka talaga", True),
        ("good game everyone", False),
        ("delete yourself noob", True),
        ("nice rotate pre", False),
        ("ul0l na to", True),
        ("GG WP", False)
    ]

    print("=" * 70)
    print(f"{Colors.BOLD}HYBRID PIPELINE INTEGRATION TESTS{Colors.RESET}")
    print("=" * 70)
    header = f"{'Message':<35} {'Expected':<10} {'Detected':<10} {'Pass'}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for msg, expected in tests:
        res = pipeline.process_timed(msg)
        got = res["flagged"]
        ok = got == expected
        symbol = "✅" if ok else "❌"
        all_ok &= ok
        print(f"{msg:<35} {str(expected):<10} {str(got):<10} {symbol}")

    print("-" * len(header))
    if all_ok:
        print(f"{Colors.GREEN}All tests passed — hybrid pipeline ready.{Colors.RESET}")
    else:
        print(f"{Colors.RED}Some tests failed — please inspect logic.{Colors.RESET}")

    # Detailed sample
    print("\nDETAILED SAMPLE:")
    example = pipeline.process_timed("ang g4g0 mo, sobrang b0b0 talaga")
    print_result(example)
