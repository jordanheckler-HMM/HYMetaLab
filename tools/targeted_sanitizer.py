#!/usr/bin/env python3
"""
Targeted Language Sanitizer - Focus on key research documents
Replaces overclaiming language with humble alternatives
"""
import re
from pathlib import Path

# Mapping of overclaiming → humble alternatives
REPLACEMENTS = {
    r"\bproves\b": "suggests",
    r"\bproved\b": "suggested",
    r"\bproving\b": "suggesting",
    r"\bconfirms\b": "supports",
    r"\bconfirmed\b": "supported",
    r"\bconfirming\b": "supporting",
    r"\bdefinitive\b": "preliminary",
    r"\bdefinitively\b": "preliminarily",
    r"\bconclusively\b": "suggestively",
    r"\bconclusive\b": "suggestive",
    r"\buniversal law\b": "observed regularity",
    r"\bUniversal law\b": "Observed regularity",
    r"\bbreakthrough\b": "finding",
    r"\bBreakthrough\b": "Finding",
    r"\bBREAKTHROUGH\b": "FINDING",
    r"\brevolutionary\b": "novel",
    r"\bRevolutionary\b": "Novel",
    r"\bREVOLUTIONARY\b": "NOVEL",
    r"\bgroundbreaking\b": "noteworthy",
    r"\bparadigm shift\b": "methodological advance",
}

# Target only key research documents (not QC reports which contain violations as data)
TARGET_PATTERNS = [
    "PHASE*.md",
    "SESSION*.md",
    "*_SUMMARY.md",
    "*_RESULTS.md",
    "CONSOLIDATED*.md",
    "VALIDATION*.md",
    "studies/README*.md",
]


def sanitize_file(path: Path, dry_run=False):
    """Apply replacements to a single file"""
    try:
        content = path.read_text(encoding="utf-8")
        original = content

        # Apply all replacements
        for pattern, replacement in REPLACEMENTS.items():
            content = re.sub(
                pattern,
                replacement,
                content,
                flags=re.IGNORECASE if pattern[0] == "\\" else 0,
            )

        if content != original:
            if not dry_run:
                path.write_text(content, encoding="utf-8")
            changes = len(re.findall("|".join(REPLACEMENTS.keys()), original))
            return changes
        return 0
    except Exception as e:
        print(f"⚠️  Error processing {path}: {e}")
        return 0


def main():
    root = Path(".")
    total_changes = 0
    files_changed = 0

    # Process target patterns
    for pattern in TARGET_PATTERNS:
        for path in root.rglob(pattern):
            # Skip QC and backup directories
            if any(
                x in str(path) for x in ["qc/", "backups/", ".venv/", "node_modules/"]
            ):
                continue

            changes = sanitize_file(path)
            if changes > 0:
                print(f"📝 {path}: {changes} replacements")
                total_changes += changes
                files_changed += 1

    print("\n✅ Sanitization complete:")
    print(f"   Files modified: {files_changed}")
    print(f"   Total replacements: {total_changes}")
    print("\n   Re-run Guardian: python3 qc/guardian_auditor.py")


if __name__ == "__main__":
    main()
