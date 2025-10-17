import pathlib
import re
import sys

bad = re.compile(r"\b(ultimate_|proof_of_ultimate)\b")
violations = []
for path in pathlib.Path(".").rglob("*.py"):
    if not any(seg in {"adapters", "studies"} for seg in path.parts):
        continue
    text = path.read_text(errors="ignore")
    if bad.search(text):
        violations.append(str(path))
if violations:
    print("ERROR: legacy modules referenced in:", *violations, sep="\n- ")
    sys.exit(1)
print("OK: no legacy modules imported in adapters/studies.")
