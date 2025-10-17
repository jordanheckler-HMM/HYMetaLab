.PHONY: dev api ui test validate package fmt lint checksums

PY=python3

dev:
	@echo "Starting API (uvicorn) and UI (streamlit) in background panes..."
	@echo "Tip: run each in separate terminals if preferred."
	@echo "API: uvicorn apps.sentinel_api:app --reload"
	@echo "UI : streamlit run apps/sentinel_ui.py"
	@echo "Or use: make api && make ui"

api:
	uvicorn apps.sentinel_api:app --reload

ui:
	streamlit run apps/sentinel_ui.py

test:
	pytest -q

validate:
	$(PY) -m tools.guardian_stub --demo

checksums:
	$(PY) -m tools.checksums dist

package: checksums
	@echo "Packaging artifacts..."
	@mkdir -p dist
	@cp -r dev/logs dist/ 2>/dev/null || true
	@$(PY) - << 'PY'
import json, os, time, hashlib, glob
from pathlib import Path
dist = Path("dist")
sumf = dist/"SHA256SUMS.txt"
lines = []
for fp in dist.rglob("*"):
    if fp.is_file() and fp.name != "SHA256SUMS.txt":
        h=hashlib.sha256(fp.read_bytes()).hexdigest()
        lines.append(f"{h}  {fp}")
sumf.write_text("\n".join(lines)+"\n")
print(f"Wrote {sumf}")
PY

fmt:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .
