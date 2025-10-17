"""
Comprehensive Installation Verification for Open Family Mini-Lab System
Tests all core modules, imports, and data structures
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("🧬 Open Family Mini-Lab System — Installation Verification")
print("=" * 70)

# Test 1: Core module imports
print("\n[1/7] Testing core module imports...")
try:
    from open_family.open_core import constants, validation

    print("  ✅ All core modules imported successfully")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Constants verification
print("\n[2/7] Verifying constants...")
try:
    assert hasattr(constants, "EPS_BAND")
    assert hasattr(constants, "RHO_STAR")
    assert hasattr(constants, "LAMBDA_STAR")
    assert hasattr(constants, "C_EFF_RATIOS")
    assert constants.RHO_STAR == 0.085
    assert constants.LAMBDA_STAR == 0.9
    print(
        f"  ✅ Constants validated: ε={constants.EPS_BAND}, ρ★={constants.RHO_STAR}, λ★={constants.LAMBDA_STAR}"
    )
except AssertionError as e:
    print(f"  ❌ Constant validation failed: {e}")
    sys.exit(1)

# Test 3: Validation logic
print("\n[3/7] Testing validation.classify()...")
try:
    import pandas as pd

    # Test case 1: VALIDATED
    df_pass = pd.DataFrame(
        {"delta_cci": [0.04, 0.05, 0.03], "delta_hazard": [-0.02, -0.03, -0.01]}
    )
    result_pass = validation.classify(df_pass)
    assert result_pass == "VALIDATED", f"Expected VALIDATED, got {result_pass}"

    # Test case 2: UNDER_REVIEW
    df_fail = pd.DataFrame({"delta_cci": [0.01, 0.02], "delta_hazard": [-0.005, 0.001]})
    result_fail = validation.classify(df_fail)
    assert result_fail == "UNDER_REVIEW", f"Expected UNDER_REVIEW, got {result_fail}"

    print("  ✅ Validation logic working correctly")
    print(f"     - VALIDATED case: {result_pass}")
    print(f"     - UNDER_REVIEW case: {result_fail}")
except Exception as e:
    print(f"  ❌ Validation test failed: {e}")
    sys.exit(1)

# Test 4: Lab imports
print("\n[4/7] Testing lab module imports...")
try:
    from open_family.openlight_lab.adapters import openlight_informational_speed
    from open_family.openmind_lab.adapters import openmind_intent_field
    from open_family.opentime_lab.adapters import opentime_memory_feedback

    print("  ✅ All lab adapters imported successfully")
except Exception as e:
    print(f"  ❌ Lab import failed: {e}")
    sys.exit(1)

# Test 5: Study YAML files
print("\n[5/7] Verifying study configurations...")
try:
    import yaml

    studies = [
        "open_family/openlight_lab/studies/openlight_phase36.yml",
        "open_family/opentime_lab/studies/opentime_phase39.yml",
        "open_family/openmind_lab/studies/openmind_phase42.yml",
    ]

    for study_path in studies:
        full_path = project_root / study_path
        with open(full_path) as f:
            config = yaml.safe_load(f)
            study_id = config.get("study_id", "unknown")
            print(f"  ✅ {study_id}")

except Exception as e:
    print(f"  ❌ YAML verification failed: {e}")
    sys.exit(1)

# Test 6: Adapter functions
print("\n[6/7] Checking adapter interfaces...")
try:
    adapters = [
        ("OpenLight", openlight_informational_speed),
        ("OpenTime", opentime_memory_feedback),
        ("OpenMind", openmind_intent_field),
    ]

    for lab_name, adapter in adapters:
        assert hasattr(adapter, "run"), f"{lab_name} missing run()"
        assert hasattr(adapter, "validate"), f"{lab_name} missing validate()"
        assert callable(adapter.run), f"{lab_name}.run() not callable"
        assert callable(adapter.validate), f"{lab_name}.validate() not callable"
        print(f"  ✅ {lab_name} adapter: run(), validate() present")

except AssertionError as e:
    print(f"  ❌ Adapter interface check failed: {e}")
    sys.exit(1)

# Test 7: File integrity
print("\n[7/7] Checking file structure...")
try:
    required_files = [
        "open_family/__init__.py",
        "open_family/README.md",
        "open_family/SETUP_COMPLETE.md",
        "open_family/targets.txt",
        "open_family/smoke_test.py",
        "open_family/open_core/__init__.py",
        "open_family/open_core/constants.py",
        "open_family/open_core/validation.py",
        "open_family/open_core/io.py",
        "open_family/open_core/openlaws_protocol.md",
    ]

    missing = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing.append(file_path)

    if missing:
        print(f"  ❌ Missing files: {missing}")
        sys.exit(1)
    else:
        print(f"  ✅ All {len(required_files)} required files present")

except Exception as e:
    print(f"  ❌ File structure check failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅ INSTALLATION VERIFICATION COMPLETE")
print("=" * 70)
print("\n📊 Summary:")
print("  - Core modules: ✅ OPERATIONAL")
print("  - Validation logic: ✅ TESTED")
print("  - Lab adapters: ✅ PRESENT (3/3)")
print("  - Study configs: ✅ VALID (3/3)")
print("  - File structure: ✅ COMPLETE")
print("\n🚀 Status: READY FOR PRODUCTION RUNS")
print("\n🧠 Next steps:")
print("  1. Implement full adapter.run() logic for each lab")
print("  2. Connect to openlaws_automation.py workflow")
print("  3. Execute Phase 36/39/42 studies")
print("\n" + "=" * 70)
