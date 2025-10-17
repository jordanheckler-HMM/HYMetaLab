---
title: API_Documentation_Attempt.md
date: 2025-10-16
version: draft
~81c7fbf233
---

# API Documentation Generation Attempt

**Date:** 2025-10-16  
**Task:** Generate HTML API documentation with pdoc  
**Status:** ⚠️ **BLOCKED BY DISK SPACE**

---

## Summary

Attempted to generate HTML API documentation for `apps/`, `tools/`, and `qc/` directories using pdoc3. The operation was blocked by insufficient disk space (only 33Mi available on a 228Gi disk that is 100% full).

---

## Command Attempted

```bash
pdoc --html --output-dir docs/api --skip-errors apps/ tools/ qc/
```

**Purpose:**
- Generate HTML documentation from Python docstrings
- Create browsable API reference
- Document all modules, classes, and functions

**Target Directories:**
- `apps/` — 6 Streamlit applications
- `tools/` — 38 utility scripts and tools
- `qc/` — 20 quality control modules

---

## Disk Space Constraint

**Current Status:**
```
Disk: 228Gi total
Used: 206Gi (90.4%)
Available: 33Mi (0.014%)
Status: 100% full
```

**Issue:** 
- Pdoc needs temporary space to import modules and generate HTML
- Only 33Mi available is insufficient
- Multiple operations failed: `[Errno 28] No space left on device`

---

## Partial Execution Results

### Import Warnings/Errors (with --skip-errors)

**Modules with Import Issues:**

1. **`apps/loop_lite_app.py`**
   - Error: `ModuleNotFoundError: No module named 'loop_lite_core'`
   - Cause: Missing dependency

2. **`tools/meaningforge_cli.py`**
   - Error: `ModuleNotFoundError: No module named 'tools.meaningforge_cli'`
   - Cause: Import path issue

3. **`tools/truthlens_cli.py`**
   - Error: `ImportError: cannot import name 'TruthLensCore' from 'truthlens_core'`
   - Cause: Missing TruthLensCore class (known issue from pytest)

4. **`tools/gen_replication_packet.py`**
   - Error: `OSError: [Errno 28] No space left on device`
   - Cause: Attempted file operation with no disk space

### Streamlit Warnings
```
Warning: Thread 'MainThread': missing ScriptRunContext!
This warning can be ignored when running in bare mode.
```
- Non-blocking, occurs when Streamlit apps are imported outside of `streamlit run`

---

## Expected Output (If Successful)

**Documentation Structure:**
```
docs/api/
├── index.html                    # Main index
├── apps/
│   ├── index.html
│   ├── guardian_check_app.html
│   ├── loop_lite_core.html
│   └── ... (4 more apps)
├── tools/
│   ├── index.html
│   ├── epistemic.html            # NEW: Our epistemic utilities
│   ├── zenodo_publisher.html
│   └── ... (36 more tools)
└── qc/
    ├── index.html
    ├── guardian_v4/
    │   └── ... (Guardian modules)
    └── ... (other QC modules)
```

**Benefits (When Generated):**
- Browsable API reference
- Searchable documentation
- Links between related functions
- Docstring rendering with formatting
- Type hints displayed
- Source code links

---

## Solutions

### Option 1: Free Disk Space (Recommended)

**Large Directories to Investigate:**
```bash
# Find large directories
du -sh * 2>/dev/null | sort -h | tail -20

# Common large directories in research projects:
- .cache/             # Python/pip cache
- outputs/            # Experiment results
- results/            # Archived results
- discovery_results/  # Discovery outputs
- __pycache__/        # Python bytecode
- .pytest_cache/      # Pytest cache
- node_modules/       # JavaScript dependencies (if any)
```

**Safe to Delete:**
```bash
# Clear Python caches
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Clear pip cache
pip cache purge

# Clear pre-commit cache
rm -rf ~/.cache/pre-commit
```

**Free Space Needed:**
- Minimum: ~100Mi for pdoc HTML generation
- Recommended: ~1Gi for comfortable operation
- Target: Free up 2-5Gi

### Option 2: Generate Documentation on CI/CD

**GitHub Actions Workflow:**
```yaml
# .github/workflows/docs.yaml
name: Documentation
on:
  push:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pip install pdoc3
      - run: pdoc --html --output-dir docs/api --skip-errors apps/ tools/ qc/
      - uses: actions/upload-artifact@v4
        with:
          name: api-docs
          path: docs/api/
      # Optional: Deploy to GitHub Pages
      - uses: peaceiris/actions-gh-pages@v4
        if: github.ref == 'refs/heads/main'
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/api
```

**Benefits:**
- No local disk space required
- Automatic updates on push
- Can deploy to GitHub Pages
- Fresh environment every time

### Option 3: Generate for Specific Modules

**Selective Documentation:**
```bash
# Document only new/important modules
pdoc --html --output-dir docs/api tools/epistemic.py
pdoc --html --output-dir docs/api apps/guardian_check_app.py
pdoc --html --output-dir docs/api qc/guardian_v4/
```

**Advantages:**
- Smaller disk space requirement
- Faster generation
- Focus on high-priority modules

### Option 4: Use Alternative Documentation Generators

**Option 4a: Sphinx (more heavyweight)**
```bash
# Requires more setup but very powerful
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs/
# Edit conf.py, index.rst
make html
```

**Option 4b: pydoc (built-in, text-only)**
```bash
# No external dependencies, minimal disk usage
python3 -m pydoc tools.epistemic > docs/api/epistemic.txt
```

**Option 4c: Markdown generation**
```bash
# Use pdoc with markdown output (smaller)
pdoc --md tools.epistemic > docs/api/epistemic.md
```

---

## Workaround: Manual Documentation

Until disk space is available, maintain manual API documentation:

**File:** `docs/API_Reference.md`

```markdown
# HYMetaLab API Reference

## tools/epistemic.py

### `BOUNDARY`
Standard epistemic boundary declaration.

### `hedge(s: str) -> str`
Add epistemic hedging to a claim.

### `wrap_claim(claim: str, confidence: str) -> str`
Wrap claim with confidence-appropriate hedging.

[... continue for key modules]
```

---

## Retry Instructions

### After Freeing Disk Space

**1. Verify Space Available:**
```bash
df -h .
# Ensure > 500Mi available
```

**2. Run pdoc:**
```bash
pdoc --html --output-dir docs/api --skip-errors apps/ tools/ qc/
```

**3. Verify Generation:**
```bash
ls -lh docs/api/
find docs/api/ -name "*.html" | wc -l
# Should see 50+ HTML files
```

**4. View Locally:**
```bash
python3 -m http.server 8000 --directory docs/api
# Visit http://localhost:8000
```

**5. Add to Git:**
```bash
git add docs/api/
git commit -m "docs: generate API documentation with pdoc"
```

---

## Known Import Issues to Fix

Before re-running pdoc, consider fixing these:

### 1. Missing `loop_lite_core` module
```python
# apps/loop_lite_app.py line 69
from loop_lite_core import compute_once, trajectory
# Either create the module or add try/except
```

### 2. Missing `TruthLensCore` class
```python
# truthlens_core.py
# Add the TruthLensCore class definition
# Or update imports in truthlens_cli.py, truthlens_validator.py
```

### 3. Missing `meaningforge_cli` module
```python
# tools/meaningforge_cli.py
# Check if file exists and imports are correct
```

**Note:** These can be addressed with `--skip-errors`, but fixing them will may improve documentation quality.

---

## Disk Space Recommendations

### Immediate Priorities

1. **Critical: Free up 2-5Gi** to enable:
   - Documentation generation
   - Pre-commit remote environments (currently using local workaround)
   - Git operations without errors
   - General development comfort

2. **Archive old results:**
   ```bash
   # Compress and move to external storage
   tar -czf archive_2025_10_16.tar.gz results/ outputs/
   # Move to external drive or cloud storage
   # Then delete originals
   ```

3. **Clear caches regularly:**
   ```bash
   # Add to weekly maintenance
   pip cache purge
   find . -type d -name "__pycache__" -delete
   find . -type d -name ".pytest_cache" -delete
   ```

### Long-term Solutions

- **External storage** for archived results
- **Cloud storage** for large datasets
- **Regular cleanup** of temporary files
- **Disk monitoring** alerts when < 10Gi free

---

## Alternative: View Documentation Without HTML

**Use pydoc server (minimal disk usage):**
```bash
# Built-in, no HTML generation
python3 -m pydoc -b
# Opens browser with searchable documentation
# No files written to disk
```

**Pros:**
- Zero disk space usage
- Works immediately
- Searchable and browsable

**Cons:**
- Less polished than pdoc HTML
- Requires Python server running
- No static files to commit

---

## Compliance Note

**SOP v1.1** recommends API documentation for all public modules. While this attempt was blocked by infrastructure constraints (disk space), the requirement remains valid and should be fulfilled once space is available.

**Workarounds in place:**
- Manual docstring documentation in code
- README files for major modules
- Operation logs documenting functionality

---

## Files Status

**Created:**
- None (blocked by disk space)

**Modified:**
- None

**Blocked:**
- `docs/api/` directory (0 files generated)

---

## Next Steps

**Immediate:**
1. Free up disk space (target: 2-5Gi)
2. Retry pdoc generation
3. Verify HTML output

**Alternative:**
1. Set up GitHub Actions for documentation
2. Deploy to GitHub Pages
3. No local disk space required

**Long-term:**
1. Regular disk space monitoring
2. Automated cleanup scripts
3. Archive strategy for old results

---

**Attempted By:** Lab Techs GPT  
**Timestamp:** 2025-10-16T15:53:00-05:00  
**Status:** Blocked by disk space (33Mi available / 228Gi total)  
**Priority:** MEDIUM (can use CI/CD or manual docs as workaround)

---

*This attempt is documented for reproducibility. The command is valid and will succeed once disk space is available.*



## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
