# Zenodo Link Guard — Automatic Status Detection

**Purpose:** Automatically detect when Zenodo DOI becomes live and update dashboard  
**Method:** Simple HTTP check via DOI resolver (no API token required)  
**Created:** 2025-10-15

---

## 🎯 Overview

The Zenodo Link Guard is a lightweight monitoring script that:
1. Checks if your Zenodo DOI is accessible via the DOI resolver
2. Automatically updates `dashboard_data.json` when it goes live
3. Logs the change to `docs/integrity/SHA256_log.md`
4. Requires no API tokens or authentication

**Why This Matters:**
- **No manual tracking** — script detects publication automatically
- **Link rot prevention** — maintains both concept and version DOIs
- **Audit trail** — logs all status changes
- **Zero config** — just paste DOI once and let it run

---

## 📋 How It Works

### Current State
```json
{
  "zenodo_concept_doi": "10.5281/zenodo.17299062",
  "zenodo_version_doi": "",
  "zenodo_status": "in_review"
}
```

### When DOI Goes Live
1. Script checks: `https://doi.org/10.5281/zenodo.17299062`
2. If HTTP 200-399 response → DOI is live
3. Automatically updates:
   ```json
   {
     "zenodo_concept_doi": "10.5281/zenodo.17299062",
     "zenodo_version_doi": "10.5281/zenodo.17299062",
     "zenodo_status": "published"
   }
   ```
4. Logs to `docs/integrity/SHA256_log.md`

---

## 🚀 Usage

### One-Time Check
```bash
python3 tools/zenodo_link_guard.py
```

### Watch Mode (Checks Every 5 Minutes)
```bash
./tools/run_zenodo_guard.sh --watch
```
Press `Ctrl+C` to stop.

### Background Mode (Daemonized)
```bash
nohup ./tools/run_zenodo_guard.sh --watch > logs/zenodo_guard.log 2>&1 &
```

### Cron Job (Checks Every 15 Minutes)
```bash
# Add to crontab
*/15 * * * * cd /path/to/repo && python3 tools/zenodo_link_guard.py >> logs/zenodo_guard.log 2>&1
```

To edit crontab:
```bash
crontab -e
```

---

## 📊 Example Output

### When DOI Is Not Yet Live
```
[2025-10-15 12:00:00 UTC] Zenodo status unchanged (ok=False, current_status=in_review).
```

### When DOI Goes Live
```
[2025-10-15 14:30:00 UTC] ✅ Zenodo status updated to published.
[2025-10-15 14:30:00 UTC] 📝 Integrity log updated.
```

---

## 🔧 Configuration

### Files Modified
- `dashboard_data.json` — Status updated to "published"
- `docs/integrity/SHA256_log.md` — Log entry added

### Files Required
- `dashboard_data.json` — Must contain `zenodo_concept_doi` or `zenodo_version_doi`

### No Configuration Needed
- No API tokens
- No credentials
- No external dependencies (uses Python stdlib)

---

## 🧪 Testing

### Manual Test (Should Return "unchanged")
```bash
python3 tools/zenodo_link_guard.py
```

### Force Status Check
```bash
# Check what the script sees
python3 -c "
import json
data = json.load(open('dashboard_data.json'))
print(f\"Concept DOI: {data.get('zenodo_concept_doi')}\")
print(f\"Version DOI: {data.get('zenodo_version_doi')}\")
print(f\"Status: {data.get('zenodo_status')}\")
"
```

### Test DOI Resolver Directly
```bash
# Should return HTTP 200 when live
curl -I https://doi.org/10.5281/zenodo.17299062
```

---

## 🎯 When To Run

### Recommended Schedule

**During Initial Upload (High-Frequency Checking):**
- Every 5 minutes for first 2 hours
- Every 15 minutes for next 6 hours
- Zenodo typically publishes within minutes to hours

**After Initial Period (Low-Frequency Maintenance):**
- Every hour or daily
- Catches any edge cases or republishing

**Command:**
```bash
# High-frequency during upload
./tools/run_zenodo_guard.sh --watch

# Or schedule in cron
*/5 * * * * cd /path/to/repo && python3 tools/zenodo_link_guard.py >> logs/zenodo_guard.log 2>&1
```

---

## 🔍 Troubleshooting

### Issue: Script Says "No DOI found"
**Solution:** Check `dashboard_data.json` has `zenodo_concept_doi` or `zenodo_version_doi`

### Issue: DOI Is Live But Status Not Updating
**Possible Causes:**
1. DOI resolver is caching
2. Network timeout
3. Permissions issue writing to `dashboard_data.json`

**Debug:**
```bash
python3 -c "
import urllib.request
url = 'https://doi.org/10.5281/zenodo.17299062'
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        print(f'Status: {r.status}')
        print(f'URL: {r.url}')
except Exception as e:
    print(f'Error: {e}')
"
```

### Issue: Script Runs But Nothing Happens
**Check:**
1. Current status: `jq '.zenodo_status' dashboard_data.json`
2. If already "published", script won't update again
3. Check logs: `tail logs/zenodo_guard.log`

---

## 🔐 Security & Privacy

**Safe:**
- ✅ No credentials required
- ✅ Read-only HTTP GET requests
- ✅ Only modifies local files (dashboard, logs)
- ✅ No data sent to external services

**What It Does:**
1. Reads `dashboard_data.json`
2. Makes HTTP request to public DOI resolver
3. Updates local files if status changes

**What It Doesn't Do:**
- ❌ No API calls to Zenodo
- ❌ No authentication
- ❌ No data upload
- ❌ No network writes

---

## 🔄 Future Enhancements

When Zenodo API token is available, upgrade to:
- Fetch exact version DOI from API
- Pull citation metadata automatically
- Verify SHA256 checksum matches
- Check for updates/new versions

**Upgrade Path:**
```python
# Future: Add API support (optional)
if zenodo_token := os.getenv("ZENODO_API_TOKEN"):
    # Use API for richer metadata
    pass
else:
    # Fallback to current resolver check
    pass
```

---

## 📚 Related Documentation

- **Zenodo Upload:** `docs/integrity/Zenodo_Upload_Instructions.md`
- **Publication Record:** `docs/integrity/Zenodo_Publication_Complete.md`
- **Integrity Log:** `docs/integrity/SHA256_log.md`
- **Dashboard:** `dashboard_data.json`

---

## 📞 Support

**Script Location:** `tools/zenodo_link_guard.py`  
**Wrapper:** `tools/run_zenodo_guard.sh`  
**Logs:** `logs/zenodo_guard.log` (if using cron/background)

**Questions?**
- Check `dashboard_data.json` for current state
- Review `docs/integrity/SHA256_log.md` for history
- Test DOI resolver: `curl -I https://doi.org/YOUR_DOI`

---

**"Integrity → Resilience → Meaning"**  
— HYMetaLab Research Charter

---

**Last Updated:** 2025-10-15
