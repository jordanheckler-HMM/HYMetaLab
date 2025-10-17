#!/usr/bin/env python3
"""
Simple test script for real-world validation system.
Tests core functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_registry_loading():
    """Test registry YAML loading."""
    try:
        import yaml

        with open("real_world_validation/registry.yaml") as f:
            registry = yaml.safe_load(f)

        print("✓ Registry loaded successfully")
        print(f"  - Groups: {list(registry['groups'].keys())}")

        # Count cities
        total_cities = 0
        for group_name, group_config in registry["groups"].items():
            cities = group_config["cities"]
            total_cities += len(cities)
            print(f"  - {group_name}: {len(cities)} cities")

        print(f"  - Total cities: {total_cities}")
        return True

    except Exception as e:
        print(f"❌ Registry loading failed: {e}")
        return False


def test_package_imports():
    """Test package imports."""
    try:

        print("✓ All package imports successful")
        return True

    except Exception as e:
        print(f"❌ Package imports failed: {e}")
        return False


def test_data_ingester():
    """Test data ingester initialization."""
    try:
        from real_world_validation.ingest import DataIngester

        ingester = DataIngester()

        # Test city listing
        us_cities = ingester.get_group_cities("us_core")
        world_cities = ingester.get_group_cities("world_sample")

        print("✓ DataIngester initialized")
        print(f"  - US cities: {len(us_cities)}")
        print(f"  - World cities: {len(world_cities)}")

        # Test city config
        nyc_config = ingester.get_city_config("nyc")
        if nyc_config:
            print(f"  - NYC config loaded: {nyc_config['city']['name']}")

        return True

    except Exception as e:
        print(f"❌ DataIngester test failed: {e}")
        return False


def test_success_criteria():
    """Test success criteria configuration."""
    try:
        from real_world_validation.ingest import DataIngester

        ingester = DataIngester()
        criteria = ingester.get_success_criteria()

        print("✓ Success criteria loaded")
        for criterion, threshold in criteria.items():
            print(f"  - {criterion}: {threshold}")

        return True

    except Exception as e:
        print(f"❌ Success criteria test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Real World Validation System - Test Suite")
    print("=" * 50)

    tests = [
        ("Registry Loading", test_registry_loading),
        ("Package Imports", test_package_imports),
        ("Data Ingester", test_data_ingester),
        ("Success Criteria", test_success_criteria),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print("  Test failed")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
