"""
Smoke Test for Open Family Mini-Lab System
Tests core validation module with synthetic data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from open_family.open_core import validation

# Create synthetic test data that meets validation thresholds
df = pd.DataFrame(
    {
        "delta_cci": [0.04, 0.05, 0.03, 0.06],
        "delta_hazard": [-0.02, -0.03, -0.01, -0.025],
    }
)

# Run classification
result = validation.classify(df)
print(f"QC: {result}")

# Verify expected result
assert result == "VALIDATED", f"Expected VALIDATED, got {result}"
print("✅ Smoke test passed: Validation system operational")

# Display summary statistics
print("\nTest Data Summary:")
print(f"  Mean ΔCCI: {df['delta_cci'].mean():.4f} (threshold: ≥0.03)")
print(f"  Mean Δhazard: {df['delta_hazard'].mean():.4f} (threshold: ≤-0.01)")
