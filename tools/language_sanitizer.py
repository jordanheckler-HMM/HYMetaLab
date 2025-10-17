import pathlib
import re

FORBIDDEN = [
    r"\bultimate\b",
    r"\bperfect\b",
    r"\b100%\s*cert(ain|ainty)\b",
    r"\bprove(d|s)\b",  # suggest "supports" / "consistent with"
]
REPLACEMENTS = {
    r"\bultimate\b": "optimized",
    r"\bperfect\b": "robust",
    r"\b100%\s*cert(ain|ainty)\b": "high confidence",
    r"\bprove(d|s)\b": "supports",
}


def sanitize_text(text):
    for pat, rep in REPLACEMENTS.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def process_file(p: pathlib.Path):
    t = p.read_text(encoding="utf-8", errors="ignore")
    new = sanitize_text(t)
    if new != t:
        p.write_text(new, encoding="utf-8")
        print("Sanitized:", p)


def main():
    root = pathlib.Path(".")
    for ext in ("*.md", "*.py", "*.yml", "*.yaml", "*.txt"):
        for p in root.rglob(ext):
            if ".venv" in p.parts or ".git" in p.parts:
                continue
            process_file(p)


if __name__ == "__main__":
    main()
