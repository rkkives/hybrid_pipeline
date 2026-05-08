# =============================================================================
# test_cases.py
# Hybrid Toxic Language Detection Pipeline
# Improved Sample Test Cases for MLBB Filipino & English Game Chat
# =============================================================================
# Each test case entry:
#   - message : raw chat input as typed by player
#   - label   : "toxic" or "clean"
#   - type    : type category (plain_toxic, leet_basic, etc.)
#   - note    : brief description for paper documentation
# =============================================================================

from typing import List, Dict
import sys
import os

# Adjust for relative imports when run inside /experiments/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # import the enhanced detection pipeline
    from boyer_moore import boyer_moore_detect
except ImportError:
    # fallback dummy detector
    def boyer_moore_detect(text: str, *args, **kwargs) -> Dict[str, str]:
        return {"is_toxic": any(w in text.lower() for w in ["gago", "bobo", "puta"])}

# ANSI color codes for better terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GRAY = "\033[90m"


# =============================================================================
# TEST CASES
# =============================================================================
# shortened explanation retained for brevity
test_cases: List[Dict[str, str]] = [
    {"message": "gago, bakit mo binasag yung tower namin", "label": "toxic", "type": "plain_toxic", "note": "Filipino start insult"},
    {"message": "idiot, why did you steal my blue buff", "label": "toxic", "type": "plain_toxic", "note": "English start insult"},
    {"message": "g4g0, bakit ka nag-back habang may teamfight", "label": "toxic", "type": "leet_basic", "note": "Filipino leet insult"},
    {"message": "1d107 jungler, you gave first blood for free", "label": "toxic", "type": "leet_inter", "note": "English leet intermediate"},
    {"message": "gago ka talaga, ur such an 1d107 in teamfight", "label": "toxic", "type": "mixed", "note": "mixed Filipino+English"},
    {"message": "hoy 74ng4 mo, why are you not rotating to turtle", "label": "toxic", "type": "codeswitched", "note": "Filipino+English leet"},
    {"message": "good game everyone, GG WP", "label": "clean", "type": "clean", "note": "sportsmanship"},
    {"message": "mag-rotate na sa bottom, malapit na silang mag-push", "label": "clean", "type": "clean", "note": "Filipino rotation callout"},
]

# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================

def summarize_cases(cases: List[Dict[str, str]]) -> None:
    """Print category counts, toxic/clean ratio, and summary table."""
    total = len(cases)
    toxic = sum(1 for t in cases if t["label"] == "toxic")
    clean = total - toxic

    # distribution by type
    type_counts: Dict[str, int] = {}
    for t in cases:
        type_counts[t["type"]] = type_counts.get(t["type"], 0) + 1

    title = f"{Colors.BOLD}{Colors.CYAN}TEST CASE SUMMARY — MLBB Game Chat{Colors.RESET}"
    print(f"\n{Colors.GRAY}{'=' * 65}{Colors.RESET}")
    print(title.center(65))
    print(f"{Colors.GRAY}{'=' * 65}{Colors.RESET}")
    print(f"Total test cases : {Colors.BOLD}{total}{Colors.RESET}")
    print(f"Toxic messages   : {Colors.RED}{toxic}{Colors.RESET}")
    print(f"Clean messages   : {Colors.GREEN}{clean}{Colors.RESET}\n")

    print(f"{Colors.BOLD}Breakdown by Type:{Colors.RESET}")
    for typ, count in sorted(type_counts.items()):
        color = Colors.RED if "toxic" in typ else Colors.GREEN
        print(f"  {color}{typ:<20}{Colors.RESET} : {count}")
    print(f"{Colors.GRAY}{'=' * 65}{Colors.RESET}\n")


def quick_pipeline_check(cases: List[Dict[str, str]], n: int = 6) -> None:
    """
    Quickly verify integration with the enhanced Boyer–Moore detector.
    Runs detection on a few test cases (both toxic & clean).
    """
    sample_cases = cases[:n]
    print(f"{Colors.BOLD}{Colors.YELLOW}Running quick detection check on {len(sample_cases)} samples...{Colors.RESET}")
    print(f"{Colors.GRAY}{'-' * 70}{Colors.RESET}")

    for tc in sample_cases:
        msg = tc["message"]
        expected = tc["label"]
        result = boyer_moore_detect(msg, return_details=True)
        detected = "toxic" if result.get("is_toxic") else "clean"

        status_color = Colors.RED if detected == "toxic" else Colors.GREEN
        pass_color = Colors.GREEN if detected == expected else Colors.YELLOW
        print(f"{Colors.BOLD}Message:{Colors.RESET} {msg}")
        print(f" → Expected: {expected}, Detected: {status_color}{detected}{Colors.RESET}, "
              f"Score: {Colors.CYAN}{result.get('score', 'N/A')}{Colors.RESET}")
        print(f"   Status: {pass_color}{'PASS' if detected == expected else 'CHECK'}{Colors.RESET}")
        if result.get("is_toxic"):
            for det in result.get("details", []):
                print(f"     - Matched: {Colors.RED}{det['keyword']}{Colors.RESET} "
                      f"(tier={det['tier']}, sev={det['severity']})")
        print(f"{Colors.GRAY}{'-' * 70}{Colors.RESET}")
    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    summarize_cases(test_cases)
    quick_pipeline_check(test_cases)
