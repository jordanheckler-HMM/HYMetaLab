# HYMetaLab Makefile — Automation Guide

**Purpose:** Streamline research workflows with one-command automation  
**Created:** 2025-10-15

---

## 🎯 Quick Start

### Publish to Zenodo (Full Automation)
```bash
# Set your API token (get from zenodo.org/account/settings/applications)
export ZENODO_API_TOKEN=your_token_here

# Publish package
make publish_zenodo
```

**What it does:**
1. ✅ Finds latest replication package
2. ✅ Creates Zenodo deposition
3. ✅ Uploads ZIP file
4. ✅ Publishes record
5. ✅ Writes back concept + version DOIs to dashboard
6. ✅ Updates integrity logs

---

## 📋 Available Targets

### `make help`
Display all available commands with examples.

### `make publish_zenodo`
**Full Zenodo publication pipeline**

**Prerequisites:**
- `ZENODO_API_TOKEN` environment variable set
- Replication package exists in `results/publication/`

**Steps:**
1. Validates API token
2. Finds latest package
3. Creates deposition with metadata
4. Uploads package
5. Publishes
6. Updates `dashboard_data.json`
7. Updates `docs/integrity/SHA256_log.md`

**Example:**
```bash
export ZENODO_API_TOKEN=abc123...
make publish_zenodo
```

**Sandbox Mode (Testing):**
```bash
export ZENODO_API_TOKEN=sandbox_token
ZENODO_SANDBOX=1 make publish_zenodo
```

---

### `make publish_zenodo_dryrun`
Preview what would be published without executing.

**Example:**
```bash
make publish_zenodo_dryrun
```

**Output:**
- Package details
- Metadata preview
- Steps that would be executed

---

### `make verify_package`
Verify replication package integrity via SHA256.

**Example:**
```bash
make verify_package
```

**Output:**
```
Package: results/publication/phase4_open_data_replication_20251015_084124.zip
SHA256 file: results/publication/phase4_open_data_replication_20251015_084124.SHA256
phase4_open_data_replication_20251015_084124.zip: OK
✅ Package integrity verified
```

---

### `make guardian_validate`
Run Guardian v4 ethics validation on synthesis narrative.

**Example:**
```bash
make guardian_validate
```

**Output:**
- Guardian alignment score
- Detailed metrics report
- Report location

---

### `make snapshot_dashboard`
Create timestamped snapshot of `dashboard_data.json`.

**Example:**
```bash
make snapshot_dashboard
```

**Output:**
```
✅ Snapshot created: results/dashboard_snapshots/dashboard_20251015_143022.json
```

---

### `make archive_runs`
Archive all validated runs with SHA256 seal.

**Example:**
```bash
make archive_runs
```

**Output:**
- Creates ZIP of `discovery_results/` and `summary_docs/`
- Generates SHA256 checksum
- Saves to `results/archive/`

---

### `make status`
Quick status overview of DOIs, package, and dashboard.

**Example:**
```bash
make status
```

**Output:**
```
📊 HYMetaLab Status
══════════════════════════════════════════════════════════════

Dashboard Status:
  Zenodo Concept DOI: 10.5281/zenodo.17299062
  Zenodo Version DOI: not set
  Zenodo Status: in_review

Latest Package:
  -rw-r--r--  384K phase4_open_data_replication_20251015_084124.zip
```

---

### `make install_deps`
Install required Python dependencies (requests, jq).

**Example:**
```bash
make install_deps
```

---

### `make clean`
Remove temporary files and Python cache.

**Example:**
```bash
make clean
```

---

## 🔑 Environment Variables

### `ZENODO_API_TOKEN` (Required for publish)
**Get your token:**
1. Visit: https://zenodo.org/account/settings/applications
2. Create new token with `deposit:write` scope
3. Copy token

**Set in terminal:**
```bash
export ZENODO_API_TOKEN=your_token_here
```

**Set permanently (add to ~/.bashrc or ~/.zshrc):**
```bash
echo 'export ZENODO_API_TOKEN=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

**Security Note:** Never commit tokens to git!

---

### `ZENODO_SANDBOX` (Optional)
Use sandbox.zenodo.org for testing.

**Example:**
```bash
ZENODO_SANDBOX=1 make publish_zenodo
```

---

## 🔄 Typical Workflows

### Workflow 1: First-Time Publication
```bash
# 1. Verify package
make verify_package

# 2. Validate with Guardian
make guardian_validate

# 3. Dry run (preview)
make publish_zenodo_dryrun

# 4. Set API token
export ZENODO_API_TOKEN=your_token

# 5. Publish (production)
make publish_zenodo

# 6. Check status
make status
```

---

### Workflow 2: Testing with Sandbox
```bash
# 1. Get sandbox token
export ZENODO_API_TOKEN=sandbox_token

# 2. Test publish to sandbox
ZENODO_SANDBOX=1 make publish_zenodo

# 3. Verify sandbox record
# Visit: sandbox.zenodo.org

# 4. When ready, publish to production
export ZENODO_API_TOKEN=production_token
make publish_zenodo
```

---

### Workflow 3: Regular Maintenance
```bash
# Check status
make status

# Create dashboard snapshot
make snapshot_dashboard

# Archive validated runs
make archive_runs

# Guardian sweep
make guardian_validate
```

---

## 📊 Output Example

### Successful Publish
```
🚀 Publishing to Zenodo...

📦 Package: results/publication/phase4_open_data_replication_20251015_084124.zip

🚀 Zenodo Publisher
   Mode: PRODUCTION
   Package: results/publication/phase4_open_data_replication_20251015_084124.zip
   Size: 0.37 MB

📝 Step 1/5: Creating deposition...
✅ Created deposition: 17299062
   Concept DOI: 10.5281/zenodo.17299062

📤 Step 2/5: Uploading package...
✅ Uploaded: phase4_open_data_replication_20251015_084124.zip (0.37 MB)

🚀 Step 3/5: Publishing deposition...
✅ Published deposition: 17299062
   Version DOI: 10.5281/zenodo.17299063

📊 Step 4/5: Updating dashboard...
✅ Updated dashboard_data.json

📝 Step 5/5: Updating integrity log...
✅ Updated docs/integrity/SHA256_log.md

╔══════════════════════════════════════════════════════════════╗
║  ✅ ZENODO PUBLICATION COMPLETE                             ║
╚══════════════════════════════════════════════════════════════╝

🔗 Concept DOI: 10.5281/zenodo.17299062
📄 Version DOI: 10.5281/zenodo.17299063
🌐 URL: https://doi.org/10.5281/zenodo.17299063

✅ Dashboard and logs updated automatically
```

---

## 🔍 Troubleshooting

### Error: "ZENODO_API_TOKEN not set"
**Solution:** Set your API token
```bash
export ZENODO_API_TOKEN=your_token_here
```

### Error: "No replication package found"
**Solution:** Generate package first
```bash
python3 tools/gen_replication_packet.py
```

### Error: "HTTP 401 Unauthorized"
**Possible causes:**
- Invalid API token
- Token expired
- Wrong token for sandbox/production

**Solution:** Regenerate token at zenodo.org/account/settings/applications

### Error: "Package integrity verification failed"
**Solution:** Regenerate SHA256
```bash
shasum -a 256 results/publication/phase4_*.zip > results/publication/phase4_*.SHA256
```

---

## 🔐 Security Best Practices

### API Token Management
1. ✅ Never commit tokens to git
2. ✅ Use environment variables
3. ✅ Rotate tokens periodically
4. ✅ Use sandbox tokens for testing
5. ✅ Revoke unused tokens

### Add to .gitignore
```bash
echo ".env" >> .gitignore
echo "*.token" >> .gitignore
```

### Use .env file (optional)
```bash
# Create .env file
echo "ZENODO_API_TOKEN=your_token" > .env

# Load in terminal
source .env

# Or use with make
env $(cat .env) make publish_zenodo
```

---

## 📚 Related Documentation

- **Zenodo Publisher:** `tools/zenodo_publisher.py`
- **Link Guard:** `tools/zenodo_link_guard.py`
- **Upload Instructions:** `docs/integrity/Zenodo_Upload_Instructions.md`
- **Publication Record:** `docs/integrity/Zenodo_Publication_Complete.md`

---

## 🎯 Future Enhancements

Planned additions:
- `make preprint_submit` — Submit to arXiv/PsyArXiv
- `make corpus_patch` — Batch patch flagged documents
- `make full_pipeline` — Run entire Phase 1-5 pipeline
- `make ci_validate` — Full CI/CD validation suite

---

**"Integrity → Resilience → Meaning"**  
— HYMetaLab Research Charter

---

**Last Updated:** 2025-10-15
