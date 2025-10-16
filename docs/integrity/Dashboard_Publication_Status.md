# Dashboard Publication Status — Zenodo Integration

**Lab Tech:** Publication  
**Last Updated:** 2025-10-15  
**Status:** ✅ NORMALIZED

---

## 🎯 Current Publication State

### Zenodo Configuration

**Concept DOI:** `10.5281/zenodo.17299062`  
**Version DOI:** `[Empty - Pending Publication]`  
**Status:** `in_review`

### Status Badge

```
┌─────────────────────────────────────┐
│  📦 Zenodo Status: IN REVIEW        │
│                                     │
│  Concept DOI:                       │
│  10.5281/zenodo.17299062            │
│                                     │
│  Version DOI: Pending publish       │
└─────────────────────────────────────┘
```

---

## 🔗 Link Integrity

### Concept DOI (Always Valid)
**URL:** https://doi.org/10.5281/zenodo.17299062  
**Status:** ✅ PERMANENT  
**Purpose:** Links to all versions of this record

**Why It's Safe:**
- Concept DOI is assigned immediately upon deposition creation
- Persists through all versions
- Never changes, even before publication
- Can be cited in papers and documentation

### Version DOI (After Publish)
**URL:** [Will be assigned on publish]  
**Status:** ⏳ PENDING  
**Purpose:** Links to this specific version (v1.0)

**When Available:**
- After clicking "Publish" in Zenodo UI
- Or after running `make publish_zenodo`
- Typically: 10.5281/zenodo.XXXXXXX (next sequential number)

---

## 📊 Dashboard Fields Explained

### Field Structure
```json
{
  "zenodo_concept_doi": "10.5281/zenodo.17299062",
  "zenodo_version_doi": "",
  "zenodo_status": "in_review"
}
```

### Field Definitions

**`zenodo_concept_doi`**
- Type: String (DOI format)
- Required: Yes
- Description: Permanent identifier for all versions
- Example: "10.5281/zenodo.17299062"
- When Set: Immediately upon deposition creation

**`zenodo_version_doi`**
- Type: String (DOI format) or Empty String
- Required: No (empty until published)
- Description: Specific version identifier
- Example: "10.5281/zenodo.17299063" (when published)
- When Set: After publication

**`zenodo_status`**
- Type: String (enum)
- Required: Yes
- Values:
  - `"draft"` — Editing, not submitted
  - `"in_review"` — Submitted, awaiting publication
  - `"published"` — Live and citable
- Current: "in_review"

---

## 🎨 Dashboard Display Logic

### Status Badge Rendering

**If `zenodo_status == "in_review"`:**
```
Status: 🟡 IN REVIEW
Show: Concept DOI (linked)
Hide: Version DOI (show "Pending publish")
Action: "Awaiting publication"
```

**If `zenodo_status == "published"`:**
```
Status: 🟢 PUBLISHED
Show: Concept DOI (linked)
Show: Version DOI (linked)
Action: "View on Zenodo"
```

**If `zenodo_status == "draft"`:**
```
Status: ⚪ DRAFT
Show: "Not yet submitted"
Action: "Complete submission"
```

---

## 🔄 State Transitions

### Normal Flow

```
draft → in_review → published
  ↓         ↓           ↓
  No DOIs   Concept     Concept + Version
            DOI only    DOIs both set
```

### Manual Transitions

**Draft → In Review:**
```bash
# Upload to Zenodo, save without publishing
jq '.zenodo_status="in_review"' dashboard_data.json > /tmp/d.json
mv /tmp/d.json dashboard_data.json
```

**In Review → Published:**
```bash
# After clicking Publish in Zenodo
jq '.zenodo_version_doi="10.5281/zenodo.XXXXXXX" | 
    .zenodo_status="published"' dashboard_data.json > /tmp/d.json
mv /tmp/d.json dashboard_data.json
```

### Automated Transitions

**Using Link Guard:**
```bash
# Automatically detects when DOI goes live
python3 tools/zenodo_link_guard.py
```

**Using Makefile:**
```bash
# Full automation (creates + publishes)
make publish_zenodo
```

---

## 🚫 Preventing Broken Links

### Problem: Empty DOI Fields
**Bad:**
```json
{
  "zenodo_doi": "",
  "zenodo_url": ""
}
```
❌ Results in broken links in dashboard

### Solution: Two-Field Model
**Good:**
```json
{
  "zenodo_concept_doi": "10.5281/zenodo.17299062",
  "zenodo_version_doi": "",
  "zenodo_status": "in_review"
}
```
✅ Always shows valid concept DOI link

### Display Logic
```javascript
// Pseudocode
if (version_doi) {
  show_link(version_doi, "View Version v1.0");
} else if (concept_doi && status == "in_review") {
  show_link(concept_doi, "View Deposit (In Review)");
} else if (concept_doi) {
  show_link(concept_doi, "View All Versions");
}
```

---

## 📝 Changelog

| Date | Status | Action | Notes |
|------|--------|--------|-------|
| 2025-10-15 | in_review | Concept DOI set | 10.5281/zenodo.17299062 |
| [Future] | published | Version DOI set | Will update automatically |

---

## 🔍 Verification Checklist

### Pre-Publication (Current State)
- ✅ Concept DOI is set and valid
- ✅ Version DOI is empty (expected)
- ✅ Status is "in_review"
- ✅ Concept DOI link works
- ✅ Dashboard shows "In Review" badge
- ✅ No broken links in dashboard
- ✅ SHA256_log.md documents status

### Post-Publication (After publish)
- ⏳ Version DOI will be set
- ⏳ Status will be "published"
- ⏳ Both DOI links will work
- ⏳ Dashboard will show "Published" badge
- ⏳ Citation will use version DOI

---

## 🔗 Related Documentation

- **Link Guard:** `tools/zenodo_link_guard.py` — Auto-detects publication
- **Publisher:** `tools/zenodo_publisher.py` — Full automation
- **Makefile:** `Makefile` — `make publish_zenodo`
- **Integrity Log:** `docs/integrity/SHA256_log.md`
- **Zenodo Guide:** `docs/integrity/Zenodo_Link_Guard_Setup.md`

---

## 📞 Support

**Current Status Query:**
```bash
make status
# or
jq '{zenodo_concept_doi, zenodo_version_doi, zenodo_status}' dashboard_data.json
```

**Manual Status Update:**
```bash
# Update to published (when ready)
jq '.zenodo_version_doi="YOUR_VERSION_DOI" | 
    .zenodo_status="published"' dashboard_data.json > /tmp/d.json
mv /tmp/d.json dashboard_data.json
```

**Verify Links:**
```bash
# Test concept DOI
curl -I https://doi.org/10.5281/zenodo.17299062

# Test version DOI (when available)
curl -I https://doi.org/10.5281/zenodo.XXXXXXX
```

---

**"Integrity → Resilience → Meaning"**  
— HYMetaLab Research Charter

---

**Certified:** Lab Tech – Publication  
**Date:** 2025-10-15  
**Status:** ✅ NORMALIZED — No broken links, proper in_review state

---

**Epistemic Boundary (OpenLaws §3.4):** Findings describe simulation-bounded behaviors within controlled model scope and do not imply universal physical laws.
