import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_report(outdir, config, summaries, figures):
    outdir = Path(outdir)
    md = []
    md.append(f"# {config.get('report',{}).get('title','Analysis Report')}\n")
    md.append(f"Author: {config.get('report',{}).get('author','Unknown')}\n")
    md.append("## Config\n")
    md.append("```json")
    md.append(json.dumps(config, indent=2))
    md.append("```\n")
    md.append("## Summaries\n")
    for k, v in summaries.items():
        md.append(f"### {k}\n")
        md.append("```")
        md.append(str(v))
        md.append("```")
    md.append("\n## Figures\n")
    for f in figures:
        md.append(f"![]({f})")

    md_path = outdir / "report" / "report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md))

    # try to convert to PDF
    try:
        import pypandoc

        pdf_path = md_path.parent / "report.pdf"
        pypandoc.convert_file(str(md_path), "pdf", outputfile=str(pdf_path))
    except Exception as e:
        logger.warning("PDF conversion failed: %s", e)
        pdf_path = None

    return md_path, pdf_path
