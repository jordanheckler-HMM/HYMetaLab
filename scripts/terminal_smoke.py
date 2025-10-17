from datetime import datetime
from pathlib import Path

out = Path(__file__).resolve().parents[1] / ".terminal_validation.txt"
out.write_text(f"terminal_ok: {datetime.utcnow().isoformat()}\n")
print(f"Wrote {out}")
