# =============================================================================
# aho_corasick.py
# Hybrid Toxic Language Detection Pipeline
# Component 3 — Aho-Corasick Algorithm
# =============================================================================
# Implements the Aho-Corasick multi-pattern string matching algorithm.
# In the hybrid pipeline, Aho-Corasick serves as the second pattern matching
# stage — a comprehensive full-dictionary scan that only runs after
# Boyer-Moore confirms a high-risk keyword in Stage 1.
#
# Design Decisions:
#   - Operates on normalized, lowercased text only (output of normalizer.py)
#   - Trie is built ONCE during initialization and reused across messages
#   - Failure links computed via BFS (Breadth-First Search)
#   - Output links ensure overlapping/nested matches are correctly reported
#   - Returns match positions consistent with Boyer-Moore output format
# =============================================================================
 
from collections import deque
 
 
# =============================================================================
# TRIE NODE
# Each node in the trie represents one character in the pattern set.
# =============================================================================
class TrieNode:
    """
    Represents a single node in the Aho-Corasick trie.
 
    Attributes:
        children     (dict) : Maps characters to child TrieNode objects
        failure_link (TrieNode): Points to longest proper suffix that is
                                 also a valid trie prefix (used on mismatch)
        output       (list) : List of complete patterns that end at this node
                              (includes patterns reachable via output links)
    """
    def __init__(self):
        self.children     = {}
        self.failure_link = None
        self.output       = []
 
 
# =============================================================================
# TRIE CONSTRUCTION
# =============================================================================
def build_trie(dictionary: list) -> TrieNode:
    """
    Builds a trie from the complete toxic word dictionary.
    Each path from root to a terminal node spells out one toxic keyword.
    Common prefixes are shared — stored only once.
 
    Parameters:
        dictionary (list): List of toxic word strings (plain text, lowercased)
 
    Returns:
        TrieNode: Root node of the constructed trie
    """
    root = TrieNode()
 
    for word in dictionary:
        word = word.lower().strip()
        if not word:
            continue
 
        current = root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
 
        # Mark end of pattern — store the full word at terminal node
        current.output.append(word)
 
    return root
 
 
# =============================================================================
# FAILURE LINK CONSTRUCTION
# =============================================================================
def build_failure_links(root: TrieNode) -> TrieNode:
    """
    Computes failure links for all trie nodes using Breadth-First Search.
 
    A failure link at node N points to the node representing the longest
    proper suffix of N's string that also exists as a prefix in the trie.
    On a mismatch during search, the algorithm follows the failure link
    instead of restarting from root — preserving linear time complexity.
 
    Output links are also merged here: each node inherits the output
    patterns of its failure link, ensuring nested/overlapping matches
    are not missed.
 
    Parameters:
        root (TrieNode): Root of the trie (output of build_trie)
 
    Returns:
        TrieNode: Same root, now with failure links and output links set
    """
    queue = deque()
 
    # Depth-1 nodes: failure link points back to root
    for char, child in root.children.items():
        child.failure_link = root
        queue.append(child)
 
    # BFS traversal — compute failure links level by level
    while queue:
        current = queue.popleft()
 
        for char, child in current.children.items():
            # Follow failure links of current node to find child's failure link
            failure = current.failure_link
 
            while failure is not None and char not in failure.children:
                failure = failure.failure_link
 
            if failure is None:
                child.failure_link = root
            else:
                next_node = failure.children.get(char)
                child.failure_link = next_node if next_node is not child else root
 
            # Merge output: child inherits matches from its failure link
            child.output = child.output + child.failure_link.output
 
            queue.append(child)
 
    return root
 
 
# =============================================================================
# CORE SEARCH — SINGLE PASS
# =============================================================================
def aho_corasick_search(text: str, root: TrieNode) -> list:
    """
    Searches for all patterns in the trie within the normalized text
    using a single linear pass. On each character:
      - If a valid transition exists, follow it
      - On mismatch, follow failure links until a valid transition is found
        or root is reached
      - At each node, report all patterns stored in output list
 
    Parameters:
        text (str)       : Normalized chat message (output of normalizer.py)
        root (TrieNode)  : Trie root with failure links built
 
    Returns:
        list: List of dicts — each containing the matched keyword
              and the start position where it was found.
 
    Example return value:
        [
            {"keyword": "gago", "positions": [4]},
            {"keyword": "bobo", "positions": [13]}
        ]
    """
    if not text:
        return []
 
    current = root
    raw_matches = []  # (start_index, keyword)
 
    for i, char in enumerate(text):
        # Follow failure links until a valid transition is found or root reached
        while current is not root and char not in current.children:
            current = current.failure_link
 
        if char in current.children:
            current = current.children[char]
 
        # Report all patterns ending at this position
        for word in current.output:
            start = i - len(word) + 1
            raw_matches.append((start, word))
 
    # Reformat to match Boyer-Moore output structure
    results = _group_matches(raw_matches)
    return results
 
 
# =============================================================================
# STANDALONE SCAN — Used for benchmarking Aho-Corasick alone
# Builds trie internally — mirrors boyer_moore_full_scan() interface
# =============================================================================
def aho_corasick_scan(text: str, dictionary: list) -> list:
    """
    Scans normalized text against a full dictionary using Aho-Corasick.
    Builds and initializes the trie internally.
 
    Used in benchmark.py to evaluate standalone Aho-Corasick performance.
 
    Parameters:
        text       (str) : Normalized chat message
        dictionary (list): Full list of toxic words from dataset
 
    Returns:
        list: All matches found — same format as boyer_moore_full_scan()
    """
    if not dictionary:
        return []
 
    root = build_trie(dictionary)
    root = build_failure_links(root)
    return aho_corasick_search(text, root)
 
 
# =============================================================================
# PREBUILT SCAN — Used by hybrid.py
# Accepts a pre-initialized trie to avoid rebuilding on every message.
# This is the function called by hybrid.py in Stage 2.
# =============================================================================
def aho_corasick_scan_prebuilt(text: str, root: TrieNode) -> list:
    """
    Scans normalized text using a pre-built Aho-Corasick trie.
    Trie is built ONCE in hybrid.py during initialization and
    passed here on every message — avoiding repeated preprocessing cost.
 
    Parameters:
        text (str)      : Normalized chat message
        root (TrieNode) : Pre-built trie with failure links
 
    Returns:
        list: All matches found
    """
    return aho_corasick_search(text, root)
 
 
# =============================================================================
# HELPER — Group raw matches by keyword
# =============================================================================
def _group_matches(raw_matches: list) -> list:
    """
    Converts a flat list of (start_index, keyword) tuples into
    the grouped dict format consistent with Boyer-Moore output.
 
    Input  : [(4, "gago"), (13, "bobo"), (4, "gago")]
    Output : [{"keyword": "gago", "positions": [4]},
              {"keyword": "bobo", "positions": [13]}]
    """
    grouped = {}
    for start, word in raw_matches:
        if word not in grouped:
            grouped[word] = []
        if start not in grouped[word]:
            grouped[word].append(start)
 
    return [{"keyword": k, "positions": v} for k, v in grouped.items()]
 
 
# =============================================================================
# QUICK TEST — run this file directly to verify Aho-Corasick is working
# =============================================================================
if __name__ == "__main__":
 
    print("=" * 65)
    print("AHO-CORASICK TEST RESULTS")
    print("=" * 65)
 
    # Build a small test dictionary
    test_dict = ["gago", "bobo", "tanga", "ulol", "inutil", "idiot", "stupid", "noob"]
    root = build_trie(test_dict)
    root = build_failure_links(root)
 
    test_cases = [
        # (normalized_text, expected_keywords_found)
        ("ang gago mo",                          ["gago"]),
        ("bobo ka talaga",                        ["bobo"]),
        ("ang gago mo bobo ka",                   ["gago", "bobo"]),
        ("wala kang silbi inutil",                ["inutil"]),
        ("good game everyone",                    []),          # clean
        ("ang tanga at bobo mo",                  ["tanga", "bobo"]),
        ("idiot ka at stupid",                    ["idiot", "stupid"]),
        ("noob ka talaga ulol",                   ["noob", "ulol"]),
        ("",                                      []),          # empty
        ("gago bobo tanga ulol inutil",           ["gago", "bobo", "tanga", "ulol", "inutil"]),
    ]
 
    all_passed = True
    print(f"{'Text':<40} {'Expected':<30} {'Pass'}")
    print("-" * 75)
 
    for text, expected_keywords in test_cases:
        result = aho_corasick_search(text, root)
        found_keywords = sorted([r["keyword"] for r in result])
        expected_sorted = sorted(expected_keywords)
        passed = found_keywords == expected_sorted
        if not passed:
            all_passed = False
        status = "✅" if passed else "❌"
        print(f"{text:<40} {str(expected_sorted):<30} {status}")
 
    print("-" * 75)
 
    # Test full scan function
    print("\nTesting aho_corasick_scan() with full dictionary:")
    print("-" * 65)
    sample = "ang gago mo, sobrang bobo talaga ng support natin"
    scan_result = aho_corasick_scan(sample, test_dict)
    print(f"Input   : '{sample}'")
    print(f"Matches : {scan_result}")
 
    print("-" * 65)
    if all_passed:
        print("All tests passed. Aho-Corasick is working correctly.")
    else:
        print("Some tests failed. Review the implementation.")
    print("=" * 65)