# HYMetaLab Reality Loop — Static Site

**Generated:** 2025-10-15  
**Version:** terminology_alignment_v1  
**Status:** ✅ LIVE

---

## 📁 Site Structure

```
site/
├── index.html              # Homepage
├── dashboard.html          # Interactive MetaDashboard
├── validation.html         # Validation reports viewer
├── replication.html        # Replication package download
├── data/
│   └── MetaDashboard_v2.json
├── validation/
│   ├── guardian_summary_v4.md
│   ├── truthlens_summary.md (optional)
│   └── meaningforge_summary.md (optional)
├── replication/
│   ├── reality_loop_packet.zip
│   └── reality_loop_packet.zip.sha256
└── assets/
    └── (optional demo videos/images)
```

---

## 🚀 Deployment

### Local Preview
```bash
# Simple Python server
cd site
python3 -m http.server 8000

# Then open: http://localhost:8000
```

### GitHub Pages
```bash
# Push to gh-pages branch
git subtree push --prefix site origin gh-pages

# Or copy site/ to your gh-pages branch
```

### Netlify / Vercel
1. Drag the `site/` folder to their web interface
2. Site will be live at your custom domain

### Static Hosting
Upload contents of `site/` to any static host:
- AWS S3 + CloudFront
- Azure Static Web Apps
- Cloudflare Pages
- Any basic web host

---

## ✅ Features

- **Responsive Design** — Works on desktop, tablet, mobile
- **Dark Mode** — Respects system preference
- **Interactive Dashboard** — Live data from JSON
- **Markdown Reports** — Dynamic loading of validation summaries
- **Integrity Verification** — SHA256 checksums for replication package
- **Zero Dependencies** — Pure HTML/CSS/JS (except marked.js CDN)
- **Fast Loading** — Minimal assets, optimized for speed

---

## 🔒 Integrity

**Site SHA256 Checksums:**
```bash
cd site
find . -type f -name "*.html" -o -name "*.json" | xargs shasum -a 256
```

**Verification:**
All data files are cryptographically sealed and logged in `docs/integrity/`.

---

## 📊 Dashboard Data Format

`data/MetaDashboard_v2.json` structure:
```json
{
  "phase4_open_data_integration": {
    "status": "COMPLETE",
    "guardian_score": 87.0,
    "validation": { "truthlens": 1.0, "meaningforge": 1.0 },
    "datasets": 5,
    "hypotheses": 5,
    "replication_package": { "size_mb": 0.37, "sha256": "..." }
  },
  "corpus_validation": { ... }
}
```

---

## 🧩 Customization

### Update Dashboard Data
Edit `data/MetaDashboard_v2.json` and refresh the page.

### Add Validation Reports
Place `.md` files in `validation/` and they'll auto-load.

### Change Theme
Edit CSS variables in `:root` of each HTML file.

---

## 📚 References

- **Zenodo DOI:** https://doi.org/10.5281/zenodo.17299062
- **Project Root:** `../` (parent directory)
- **Terminology Map:** `../docs/research/terminology_map.md`
- **Guardian Calibration:** `../docs/qc/guardian_scoring.md`

---

**"Integrity → Resilience → Meaning"**  
— HYMetaLab Research Charter
