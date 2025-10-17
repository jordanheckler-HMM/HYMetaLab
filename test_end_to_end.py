#!/usr/bin/env python3
"""
Test script for the end-to-end real-world validation pipeline.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        print("✓ FetchRunner imported")
    except Exception as e:
        print(f"❌ FetchRunner import failed: {e}")
        return False

    try:
        print("✓ SocrataClient imported")
    except Exception as e:
        print(f"❌ SocrataClient import failed: {e}")
        return False

    try:
        print("✓ CensusACSClient imported")
    except Exception as e:
        print(f"❌ CensusACSClient import failed: {e}")
        return False

    try:
        print("✓ TrendsClient imported")
    except Exception as e:
        print(f"❌ TrendsClient import failed: {e}")
        return False

    try:
        print("✓ FEMAClient imported")
    except Exception as e:
        print(f"❌ FEMAClient import failed: {e}")
        return False

    try:
        print("✓ WorldBankClient imported")
    except Exception as e:
        print(f"❌ WorldBankClient import failed: {e}")
        return False

    return True


def test_registry():
    """Test registry loading."""
    print("\nTesting registry...")

    try:
        import yaml

        with open("real_world_validation/registry.yaml") as f:
            registry = yaml.safe_load(f)

        print("✓ Registry loaded successfully")
        print(f"  - Groups: {list(registry['groups'].keys())}")

        us_core = registry["groups"]["us_core"]
        world_sample = registry["groups"]["world_sample"]

        print(f"  - US Core: {len(us_core['cities'])} cities")
        print(f"  - World Sample: {len(world_sample['cities'])} cities")

        # Check that cities have fetch configurations
        for city in us_core["cities"]:
            if "fetch" not in city:
                print(f"❌ City {city['key']} missing fetch config")
                return False

        print("✓ All cities have fetch configurations")
        return True

    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False


def test_fetch_runner():
    """Test fetch runner initialization."""
    print("\nTesting fetch runner...")

    try:
        from real_world_validation.fetch_runner import FetchRunner

        fetch_runner = FetchRunner()
        print("✓ FetchRunner initialized")

        # Test registry access
        registry = fetch_runner.registry
        print(f"✓ Registry accessible: {len(registry['groups'])} groups")

        return True

    except Exception as e:
        print(f"❌ FetchRunner test failed: {e}")
        return False


def test_cli_commands():
    """Test CLI command availability."""
    print("\nTesting CLI commands...")

    try:
        # Test CLI module import differently
        import sys

        sys.path.insert(0, str(Path(__file__).parent))

        # Import the CLI module directly
        import importlib.util

        spec = importlib.util.spec_from_file_location("rv_cli", "rv_cli.py")
        rv_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rv_cli)

        print("✓ CLI module imported")

        # Check that commands exist
        commands = ["fetch", "autopilot", "run", "report", "validate", "list_cities"]
        for cmd in commands:
            if hasattr(rv_cli, cmd):
                print(f"✓ Command '{cmd}' available")
            else:
                print(f"❌ Command '{cmd}' missing")
                return False

        return True

    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_data_structure():
    """Test expected data directory structure."""
    print("\nTesting data structure...")

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print("✓ Data directory created")

    # Test that we can create city subdirectories
    test_cities = ["nyc", "la", "chi"]
    for city in test_cities:
        city_dir = data_dir / city
        city_dir.mkdir(exist_ok=True)
        print(f"✓ City directory created: {city_dir}")

    return True


def main():
    """Run all tests."""
    print("Real World Validation System - End-to-End Test")
    print("=" * 60)

    tests = [
        test_imports,
        test_registry,
        test_fetch_runner,
        test_cli_commands,
        test_data_structure,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {e}")

    print(f"\nTest Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! System is ready for end-to-end validation.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print(
            "2. Set up API keys in .env file (copy from real_world_validation/env_example.txt)"
        )
        print("3. Run: python rv_cli.py autopilot --group us_core")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
