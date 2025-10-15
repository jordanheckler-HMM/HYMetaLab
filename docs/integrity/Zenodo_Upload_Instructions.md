# Zenodo Upload Instructions — Phase 4 Replication Package

**Lab Tech:** Archival  
**Date:** 2025-10-15  
**Package:** `phase4_open_data_replication_20251015_084124.zip`

---

## 📤 Step-by-Step Zenodo Upload

### **1. Create/Login to Zenodo Account**
- Visit: https://zenodo.org
- Create account (free) or login
- Verify email address

### **2. Create New Upload**
- Click "Upload" → "New Upload"
- Choose "Upload" type (not "New Version")

### **3. Upload Files**
- **Main file:** `phase4_open_data_replication_20251015_084124.zip` (384 KB)
- **Checksum:** `SHA256.txt` (128 bytes)
- Drag and drop or click "Choose files"

### **4. Fill Metadata Form**

#### **Basic Information:**
- **Upload type:** Dataset
- **Publication date:** 2025-10-15
- **Title:** 
  ```
  Open-Source Data Integration Pipeline — Phase 4 Replication Package
  ```

#### **Creators/Authors:**
- **Name:** HYMetaLab / Heck Yeah Simulation Research Initiative
- **Affiliation:** HYMetaLab
- **ORCID:** [If available]

#### **Description:**
```
Open-Source Data Integration Pipeline (Phase 4)
HYMetaLab / Heck Yeah Simulation Research Initiative

Complete replication package for open-source data integration research, including:
- 5 standardized open-source datasets (WVS, OECD, GSS, ESS, Cooperative Learning)
- 5 preregistered hypotheses examining trust, well-being, collaboration, and resilience
- Complete analysis code (ETL pipeline, validation tools, synthesis scripts)
- Guardian v4 ethical validation (87.0/100 PASS)
- SHA256 integrity seals for all data files

This package enables full reproduction of the Phase 4 Open-Source Data Integration study, 
which synthesized testable hypotheses from five internationally recognized datasets using 
the HYMetaLab validation pipeline (TruthLens, MeaningForge, Guardian v4).

See REPRODUCTION.md (included in ZIP) for complete replication instructions.

Validation Scores:
- TruthLens: 1.000/1.000 (perfect preregistration quality)
- MeaningForge: 1.000/1.000 (perfect semantic coherence)
- Guardian v4: 87.0/100 (ethical alignment PASS)

Statistical Methods: Pearson correlation, linear regression, bootstrap CI (n=1000)
Seeds: 11, 17, 23, 29 (deterministic, reproducible)

SHA256 Checksum:
a33e79978d2ee00b3089eae7c6be0400f8c0f4d4aabf9d1002a2d100d4c0cf02

For issues or questions, see GitHub repository (link in related identifiers).
```

#### **Keywords:**
```
open-source data
data integration
trust
well-being
collaboration
resilience
preregistration
reproducible research
ethical AI
Guardian validation
TruthLens
MeaningForge
CCI framework
Universal Resilience Law
```

#### **Additional Descriptions:**
- **Methods:** Data integration pipeline with ethical validation
- **Technical Info:** Python 3.9+, pandas, numpy, scikit-learn, pyyaml

#### **License:**
- **Recommended:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Allows:** Commercial use, modification, distribution with attribution

#### **Related Identifiers (if available):**
- **GitHub:** [Your repository URL]
- **Preprint:** [When available]
- **Related datasets:** WVS, OECD, GSS, ESS DOIs (if applicable)

#### **Contributors:**
- **Lab Tech – Validation:** Validation and quality control
- **Lab Tech – Archival:** Package preparation and integrity verification
- **Lab Tech – Integration:** Pipeline development and execution

#### **Funding:**
- [If applicable]

#### **References:**
- World Values Survey Wave 7: https://www.worldvaluessurvey.org
- OECD Education at a Glance: https://doi.org/10.1787/eag-2023-en
- General Social Survey: https://gss.norc.org
- European Social Survey: https://doi.org/10.21338/ESS10

### **5. Choose Communities (Optional but Recommended)**
- Open Science
- Social Sciences
- Research Data
- Reproducibility

### **6. Review and Publish**
- Check all fields
- Preview DOI: `10.5281/zenodo.[assigned-number]`
- Click "Publish"
- **⚠️ WARNING:** Publishing is permanent (but new versions can be created)

### **7. After Publication**
- **Copy DOI:** e.g., `https://doi.org/10.5281/zenodo.1234567`
- **Update dashboard_data.json:** Paste DOI in appropriate field
- **Update SHA256_log.md:** Add DOI to publication status
- **Update OPEN_DATA_INTEGRATION_HANDOFF.md:** Reference Zenodo DOI
- **Share:** Tweet, email, GitHub README

---

## 📋 Post-Upload Checklist

- [ ] Zenodo upload completed successfully
- [ ] DOI assigned and recorded
- [ ] DOI badge added to README.md
- [ ] dashboard_data.json updated with DOI
- [ ] SHA256_log.md updated with publication status
- [ ] Preprint submission includes Zenodo DOI
- [ ] Team notified of publication
- [ ] Archive snapshot of MetaDashboard

---

## 🔍 Verification After Upload

**Test the Package:**
```bash
# Download from Zenodo
wget https://zenodo.org/record/[YOUR-ID]/files/phase4_open_data_replication_20251015_084124.zip

# Verify checksum
shasum -a 256 phase4_open_data_replication_20251015_084124.zip
# Should match: a33e79978d2ee00b3089eae7c6be0400f8c0f4d4aabf9d1002a2d100d4c0cf02

# Extract and test
unzip phase4_open_data_replication_20251015_084124.zip
cd phase4_open_data_replication_20251015_084124
cat REPRODUCTION.md
```

---

## 📞 Support

**Zenodo Help:** https://help.zenodo.org  
**DOI Questions:** support@zenodo.org  
**Lab Contact:** [Your lab email]

---

*"Integrity → Resilience → Meaning"*  
— HYMetaLab Research Charter
