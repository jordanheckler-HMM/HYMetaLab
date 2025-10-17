import pathlib
import re
import sys

bad = re.compile(r"\b(ultimate_|proof_of_ultimate)\b")
viol = []
for base in ("adapters", "studies"):
    if not pathlib.Path(base).exists():
        continue
    for p in pathlib.Path(base).rglob("*"):
        if p.suffix not in (".py", ".yml", ".yaml"):
            continue
        try:
            txt = p.read_text(errors="ignore")
        except:
            continue
        if bad.search(txt):
            viol.append(str(p))
if viol:
    print("ERROR: legacy/demo modules referenced:\n- " + "\n- ".join(viol))
    sys.exit(1)
print("OK: no legacy modules referenced in adapters/studies.")
import pathlib
import re
import sys

bad = re.compile(r"\b(ultimate_|proof_of_ultimate)\b")
viol = []
for p in pathlib.Path("adapters").rglob("*.py"):
    if bad.search(p.read_text(errors="ignore")):
        viol.append(str(p))
for p in pathlib.Path("studies").rglob("*.yml"):
    if bad.search(p.read_text(errors="ignore")):
        viol.append(str(p))
if viol:
    print("ERROR: legacy modules referenced:\n- " + "\n- ".join(viol))
    sys.exit(1)
print("OK: no legacy modules in adapters/studies.")
