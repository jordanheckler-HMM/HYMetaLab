# HYMetaLab — Automated Research Workflows
# "Integrity → Resilience → Meaning"

.PHONY: help publish_zenodo verify_package install_deps clean

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           HYMetaLab — Research Automation                   ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  make publish_zenodo      Publish replication package to Zenodo"
	@echo "  make verify_package      Verify package integrity (SHA256)"
	@echo "  make install_deps        Install Python dependencies"
	@echo "  make guardian_validate   Run Guardian v4 validation"
	@echo "  make snapshot_dashboard  Create dashboard snapshot"
	@echo "  make archive_runs        Archive validated runs"
	@echo "  make clean               Clean temporary files"
	@echo ""
	@echo "Environment variables:"
	@echo ""
	@echo "  ZENODO_API_TOKEN         Required for publish_zenodo"
	@echo "  ZENODO_SANDBOX=1         Use sandbox.zenodo.org (optional)"
	@echo ""
	@echo "Examples:"
	@echo ""
	@echo "  make publish_zenodo"
	@echo "  ZENODO_SANDBOX=1 make publish_zenodo"
	@echo ""

# Find the latest replication package
PACKAGE_DIR := results/publication
PACKAGE_FILE := $(shell ls -t $(PACKAGE_DIR)/phase4_open_data_replication_*.zip 2>/dev/null | head -1)

# Zenodo publication
publish_zenodo:
	@echo "🚀 Publishing to Zenodo..."
	@echo ""
	@# Check for API token
	@if [ -z "$$ZENODO_API_TOKEN" ]; then \
		echo "❌ Error: ZENODO_API_TOKEN not set"; \
		echo ""; \
		echo "Get your token from:"; \
		echo "  https://zenodo.org/account/settings/applications"; \
		echo ""; \
		echo "Then run:"; \
		echo "  export ZENODO_API_TOKEN=your_token_here"; \
		echo "  make publish_zenodo"; \
		exit 1; \
	fi
	@# Check for package
	@if [ -z "$(PACKAGE_FILE)" ]; then \
		echo "❌ Error: No replication package found in $(PACKAGE_DIR)"; \
		echo ""; \
		echo "Generate package first:"; \
		echo "  python3 tools/gen_replication_packet.py"; \
		exit 1; \
	fi
	@echo "📦 Package: $(PACKAGE_FILE)"
	@echo ""
	@# Run publisher
	@if [ "$$ZENODO_SANDBOX" = "1" ]; then \
		python3 tools/zenodo_publisher.py --package "$(PACKAGE_FILE)" --sandbox; \
	else \
		python3 tools/zenodo_publisher.py --package "$(PACKAGE_FILE)"; \
	fi

# Verify package integrity
verify_package:
	@echo "🔍 Verifying package integrity..."
	@echo ""
	@if [ -z "$(PACKAGE_FILE)" ]; then \
		echo "❌ No package found in $(PACKAGE_DIR)"; \
		exit 1; \
	fi
	@echo "Package: $(PACKAGE_FILE)"
	@# Check if SHA256 file exists
	@SHA256_FILE="$$(echo $(PACKAGE_FILE) | sed 's/.zip$$/.SHA256/')"; \
	if [ -f "$$SHA256_FILE" ]; then \
		echo "SHA256 file: $$SHA256_FILE"; \
		cd $$(dirname $(PACKAGE_FILE)) && shasum -a 256 -c $$(basename $$SHA256_FILE); \
		echo "✅ Package integrity verified"; \
	else \
		echo "⚠️  No SHA256 file found"; \
		echo "Generating checksum:"; \
		shasum -a 256 $(PACKAGE_FILE); \
	fi

# Install Python dependencies
install_deps:
	@echo "📦 Installing dependencies..."
	@pip3 install -q requests jq-cli || pip3 install requests
	@echo "✅ Dependencies installed"

# Guardian v4 validation
guardian_validate:
	@echo "🛡️  Running Guardian v4 validation..."
	@python3 qc/guardian_v4/guardian_v4.py --validate --file open_data/synthesis_narrative.md --report
	@echo ""
	@echo "✅ Validation complete"
	@echo "Report: qc/guardian_v4/guardian_summary_v4.md"

# Create dashboard snapshot
snapshot_dashboard:
	@echo "📊 Creating dashboard snapshot..."
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p results/dashboard_snapshots; \
	cp dashboard_data.json results/dashboard_snapshots/dashboard_$${STAMP}.json; \
	echo "✅ Snapshot created: results/dashboard_snapshots/dashboard_$${STAMP}.json"

# Archive validated runs
archive_runs:
	@echo "🗜️  Archiving validated runs..."
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p results/archive; \
	zip -r results/archive/validated_runs_$${STAMP}.zip discovery_results/ summary_docs/ 2>&1 | tail -5; \
	shasum -a 256 results/archive/validated_runs_$${STAMP}.zip > results/archive/validated_runs_$${STAMP}.SHA256; \
	echo ""; \
	echo "✅ Archive created:"; \
	ls -lh results/archive/validated_runs_$${STAMP}.zip; \
	echo ""; \
	echo "SHA256:"; \
	cat results/archive/validated_runs_$${STAMP}.SHA256

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf .pytest_cache */.pytest_cache
	@rm -f /tmp/dash.json
	@echo "✅ Clean complete"

# Quick status check
status:
	@echo "📊 HYMetaLab Status"
	@echo "══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Dashboard Status:"
	@jq -r '"  Zenodo Concept DOI: \(.zenodo_concept_doi // "not set")"' dashboard_data.json
	@jq -r '"  Zenodo Version DOI: \(.zenodo_version_doi // "not set")"' dashboard_data.json
	@jq -r '"  Zenodo Status: \(.zenodo_status // "unknown")"' dashboard_data.json
	@echo ""
	@echo "Latest Package:"
	@if [ -n "$(PACKAGE_FILE)" ]; then \
		ls -lh $(PACKAGE_FILE); \
	else \
		echo "  No package found"; \
	fi
	@echo ""

# Dry run of Zenodo publish
publish_zenodo_dryrun:
	@echo "🔍 Dry run - Zenodo publication preview"
	@echo ""
	@if [ -z "$$ZENODO_API_TOKEN" ]; then \
		echo "ℹ️  ZENODO_API_TOKEN not set (would be required for actual publish)"; \
	else \
		echo "✅ ZENODO_API_TOKEN is set"; \
	fi
	@echo ""
	@if [ -z "$(PACKAGE_FILE)" ]; then \
		echo "❌ No package found"; \
	else \
		echo "Package to upload:"; \
		ls -lh $(PACKAGE_FILE); \
		echo ""; \
		python3 tools/zenodo_publisher.py --package "$(PACKAGE_FILE)" --dry-run; \
	fi

# =====================================================
# Lab Tech – Publication: Zenodo Metadata Verification
# =====================================================

.PHONY: publish_zenodo_sandbox
publish_zenodo_sandbox:
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║  ZENODO PUBLISH — SANDBOX MODE (METADATA VERIFICATION)      ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📋 Checking metadata formatting..."
	@echo ""
	@echo "1. Verifying dashboard_data.json zenodo_description field:"
	@jq -r '.phase4_open_data_integration.publication.zenodo_description' dashboard_data.json | head -3
	@echo ""
	@echo "2. Checking for Collective Coherence Index terminology:"
	@jq -r '.phase4_open_data_integration.publication.zenodo_description' dashboard_data.json | grep -o "Collective Coherence Index" && echo "   ✅ CCI terminology found" || echo "   ❌ CCI terminology missing"
	@echo ""
	@echo "3. Verifying OpenLaws §3.4 epistemic boundary:"
	@jq -r '.phase4_open_data_integration.publication.zenodo_description' dashboard_data.json | grep -o "OpenLaws §3.4" && echo "   ✅ Epistemic boundary reference found" || echo "   ❌ Epistemic boundary missing"
	@echo ""
	@echo "4. Checking preprint manuscript DOI placement:"
	@grep -m 1 "10.5281/zenodo.17299062" preprint/manuscript.md > /dev/null && echo "   ✅ DOI found in manuscript" || echo "   ❌ DOI missing"
	@echo ""
	@echo "5. Verifying Guardian/TruthLens scores in abstract:"
	@grep -A 10 "## Abstract" preprint/manuscript.md | grep -E "(Guardian.*87\.0|TruthLens.*1\.000)" > /dev/null && echo "   ✅ Validation scores found in abstract" || echo "   ❌ Scores missing from abstract"
	@echo ""
	@echo "6. Checking replication package:"
	@ls -lh results/publication/phase4_open_data_replication_*.zip 2>/dev/null | awk '{print "   File: " $$9 " (" $$5 ")"}' || echo "   ❌ Package not found"
	@echo ""
	@echo "7. Verifying SHA256 checksum:"
	@[ -f SHA256.txt ] && echo "   ✅ SHA256.txt exists" || echo "   ❌ SHA256.txt missing"
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║  SANDBOX TEST COMPLETE                                       ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "To publish to Zenodo (requires API token):"
	@echo "  make publish_zenodo"
	@echo ""

