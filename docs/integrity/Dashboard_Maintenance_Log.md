# MetaDashboard Maintenance Log

**Maintained by:** Lab Tech – Dashboard Maintenance  
**Purpose:** Track dashboard snapshots and archival events

---

## Snapshot History

### 2025-10-15 — Phase 4 Open Data Integration Complete

**Snapshot File:** `results/dashboard_snapshots/dashboard_20251015_HHMMSS.json`  
**Git Tag:** `dashboard-20251015_HHMMSS`  
**Commit Message:** MetaDashboard snapshot 20251015_HHMMSS

**Key Metrics at Snapshot:**
- **Phase 4 Status:** COMPLETE
- **Guardian Score:** 87.0/100 (PASS)
- **Validation:**
  - TruthLens: 1.000
  - MeaningForge: 1.000
  - Guardian v4: 87.0
- **Datasets:** 5 integrated
- **Hypotheses:** 5 generated
- **Predictions:** 20 testable

**Replication Package:**
- File: `phase4_open_data_replication_20251015_084124.zip`
- Size: 0.37 MB
- SHA256: `a33e79978d2ee00b3089eae7c6be0400f8c0f4d4aabf9d1002a2d100d4c0cf02`
- Status: Verified and ready for Zenodo

**Corpus Validation:**
- Documents Scanned: 601
- Mean Score: 62.5/100
- Passing Rate: 7.5%
- Flagged: 601 (100%)

**Publication Status:**
- Zenodo DOI: PENDING
- Preprint: PENDING

**Actions Logged:**
1. ✅ Dashboard snapshot created
2. ✅ Git commit created
3. ✅ Git tag applied
4. ⏳ Awaiting Zenodo upload
5. ⏳ Awaiting DOI assignment

---

## Snapshot Schedule

**Frequency:** Ad-hoc (after major milestones)  
**Triggered by:**
- Phase completions
- Major validation events
- Publication milestones
- Quarterly reviews

**Next Scheduled Snapshot:** After Zenodo DOI assignment

---

## Retrieval Instructions

### **View Snapshot:**
```bash
cat results/dashboard_snapshots/dashboard_YYYYMMDD_HHMMSS.json
```

### **List All Snapshots:**
```bash
ls -lh results/dashboard_snapshots/
```

### **View Tagged Commits:**
```bash
git tag --list 'dashboard-*'
```

### **Restore from Snapshot:**
```bash
# View specific snapshot
git show dashboard-YYYYMMDD_HHMMSS:results/dashboard_snapshots/dashboard_YYYYMMDD_HHMMSS.json

# Restore to working directory
cp results/dashboard_snapshots/dashboard_YYYYMMDD_HHMMSS.json dashboard_data.json
```

### **Compare Snapshots:**
```bash
diff results/dashboard_snapshots/dashboard_OLD.json results/dashboard_snapshots/dashboard_NEW.json
```

---

## Maintenance Procedures

### **Creating Manual Snapshot:**
```bash
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results/dashboard_snapshots
cp dashboard_data.json results/dashboard_snapshots/dashboard_${STAMP}.json
git add results/dashboard_snapshots/dashboard_${STAMP}.json
git commit -m "MetaDashboard snapshot ${STAMP}"
git tag dashboard-${STAMP}
```

### **Archiving Old Snapshots:**
```bash
# Move snapshots older than 1 year to archive
find results/dashboard_snapshots/ -name "*.json" -mtime +365 -exec mv {} results/archive/ \;
```

### **Verifying Snapshot Integrity:**
```bash
# Verify JSON structure
python3 -m json.tool results/dashboard_snapshots/dashboard_YYYYMMDD_HHMMSS.json > /dev/null
echo "Snapshot integrity: OK"
```

---

## Snapshot Retention Policy

**Active Snapshots (results/dashboard_snapshots/):**
- Keep all snapshots from current year
- Keep quarterly snapshots from previous 2 years
- Keep annual snapshots indefinitely

**Archived Snapshots (results/archive/):**
- Compressed snapshots older than 1 year
- Indexed in archive manifest

**Git Tags:**
- Never delete tags (permanent record)
- Tags provide audit trail

---

## Audit Trail

| Date | Snapshot | Git Tag | Trigger | Lab Tech |
|------|----------|---------|---------|----------|
| 2025-10-15 | dashboard_20251015_*.json | dashboard-20251015_* | Phase 4 Complete | Dashboard Maintenance |

---

*"Integrity → Resilience → Meaning"*  
— HYMetaLab Research Charter
