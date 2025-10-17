#!/bin/bash
# Reproducibility Script v2: Integrity Addendum Implementation
# Runs all validation tests and experiments with proper error handling

set -e  # Exit on any error

echo "🚀 Starting Reproducibility Script v2 - Integrity Addendum"
echo "=========================================================="

# Set up environment
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="discovery_results/v2_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Function to run command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    local log_file="$OUTPUT_DIR/${3:-$(echo "$description" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')}.log"
    
    echo "🔧 $description"
    echo "   Command: $cmd"
    
    if eval "$cmd" > "$log_file" 2>&1; then
        echo "   ✅ Success"
    else
        echo "   ❌ Failed (see $log_file)"
        echo "   Last 10 lines of error log:"
        tail -10 "$log_file"
        return 1
    fi
    echo ""
}

# 1. CCI Math Tests
echo "1️⃣ Running CCI Math Tests"
run_command "python -m pytest tests/test_cci_math.py -v" "CCI Math Unit Tests" "cci_tests"

# 2. Fear-Violence Integration Tests  
echo "2️⃣ Running Fear-Violence Integration Tests"
run_command "python -m pytest tests/test_fear_hooks.py -v" "Fear-Violence Integration Tests" "fear_tests"

# 3. Social Weight Reconciliation
echo "3️⃣ Running Social Weight Reconciliation"
run_command "python scripts/reconcile_social_weight.py" "Social Weight Reconciliation" "social_weight_reconciliation"

# 4. Validation Pipeline with Time Scale Alignment
echo "4️⃣ Running Validation Pipeline with Time Scale Alignment"
run_command "python rv_cli.py validate --verbose" "Validation Pipeline Check" "validation_pipeline"

# 5. Fear-Violence Experiments Repair
echo "5️⃣ Running Fear-Violence Experiments Repair"
run_command "python scripts/repair_fear_violence_experiments.py" "Fear-Violence Experiments Repair" "fear_violence_repair"

# 6. Regrowth Ablation Study
echo "6️⃣ Running Regrowth Ablation Study"
run_command "python themes/theme2_shocks/ablate_regrowth.py" "Regrowth Parameter Ablation" "regrowth_ablation"

# 7. Generate Integrity Report
echo "7️⃣ Generating Integrity Report"
cat > "$OUTPUT_DIR/integrity_report.md" << EOF
# Integrity Addendum v2 - Execution Report

**Date:** $(date)
**Status:** COMPLETED ✅
**Output Directory:** $OUTPUT_DIR

## Tests Executed

### ✅ CCI Math Tests
- Unit tests for epsilon guards and monotonicity
- Log: cci_tests.log

### ✅ Fear-Violence Integration Tests  
- Validation of fear updates in shock/inequality scenarios
- Log: fear_tests.log

### ✅ Social Weight Reconciliation
- Cross-module parameter consistency validation
- Log: social_weight_reconciliation.log

### ✅ Validation Pipeline
- Time scale alignment checks
- Log: validation_pipeline.log

### ✅ Fear-Violence Experiments Repair
- Re-run experiments with fixed adapters
- Log: fear_violence_repair.log

### ✅ Regrowth Ablation Study
- Statistical analysis of regrowth parameter independence
- Log: regrowth_ablation.log

## Files Generated

EOF

# List generated files
find "$OUTPUT_DIR" -name "*.csv" -o -name "*.json" -o -name "*.png" -o -name "*.md" | while read file; do
    echo "- $(basename "$file")" >> "$OUTPUT_DIR/integrity_report.md"
done

echo "📊 Integrity Report generated: $OUTPUT_DIR/integrity_report.md"

# 8. Summary Statistics
echo "8️⃣ Generating Summary Statistics"
cat > "$OUTPUT_DIR/summary_stats.json" << EOF
{
    "execution_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "output_directory": "$OUTPUT_DIR",
    "tests_executed": 6,
    "all_tests_passed": true,
    "files_generated": $(find "$OUTPUT_DIR" -type f | wc -l),
    "total_size_mb": $(du -sm "$OUTPUT_DIR" | cut -f1)
}
EOF

# 9. Create Makefile target
echo "9️⃣ Creating Makefile Integration"
cat >> Makefile << EOF

# Integrity Addendum v2 targets
.PHONY: v2-test v2-repro v2-clean

v2-test:
	@echo "Running Integrity Addendum v2 tests..."
	@bash scripts/repro_v2.sh

v2-repro:
	@echo "Running full reproducibility suite v2..."
	@bash scripts/repro_v2.sh
	@echo "✅ All v2 tests completed successfully"

v2-clean:
	@echo "Cleaning v2 outputs..."
	@rm -rf discovery_results/v2_*
	@echo "✅ v2 outputs cleaned"
EOF

echo ""
echo "🎉 Reproducibility Script v2 Completed Successfully!"
echo "=================================================="
echo ""
echo "📁 All outputs saved to: $OUTPUT_DIR"
echo "📊 Summary statistics: $OUTPUT_DIR/summary_stats.json"
echo "📋 Integrity report: $OUTPUT_DIR/integrity_report.md"
echo ""
echo "🔧 Available make targets:"
echo "   make v2-test    - Run all v2 tests"
echo "   make v2-repro   - Run full reproducibility suite"
echo "   make v2-clean   - Clean v2 outputs"
echo ""
echo "✅ All Integrity Addendum v2 objectives completed!"
