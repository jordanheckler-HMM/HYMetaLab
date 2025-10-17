"""Scan discovery_results and test outputs for provenance.json files that indicate used_mocks==True.
Exit with non-zero status if any such file is found. Intended for CI to fail when mock outputs are produced.
"""

import json
import sys
from pathlib import Path

roots = [Path("discovery_results"), Path(".")]
found = []
for r in roots:
    for p in r.rglob("provenance.json"):
        try:
            j = json.loads(p.read_text())
            if j.get("used_mocks"):
                found.append(str(p))
        except Exception:
            continue

if found:
    print("Found provenance.json files indicating used_mocks==True:")
    for f in found:
        print(" -", f)
    print("\nFailing CI because mock artifacts were produced.\n")
    sys.exit(2)
else:
    print("No mock provenance found.")
    sys.exit(0)
