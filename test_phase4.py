#!/usr/bin/env python3
"""
Phase 4 Test Script

Quick test of the complete Phase 4 implementation without running full simulations.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_phase4_modules():
    """Test all Phase 4 modules."""
    print("Testing Phase 4 Multi-Scale & Validation Framework")
    print("=" * 60)

    # Test observation adapters
    print("\n1. Testing Observation Adapters...")
    try:
        from phase4.observation_adapters import (
            load_all_observations,
            load_clinical_km,
            load_epi_series,
            load_fairness_series,
        )

        # Test loading observations
        observations = load_all_observations("./observations")
        print(f"   ✅ Loaded {len(observations['clinical'])} clinical datasets")
        print(f"   ✅ Loaded {len(observations['epidemic'])} epidemic datasets")
        print(f"   ✅ Loaded {len(observations['fairness'])} fairness datasets")

        # Test individual loaders
        clinical_data = load_clinical_km("./observations/clinical_km_oncology.csv")
        print(f"   ✅ Clinical KM data: {len(clinical_data.times)} time points")

        epi_data = load_epi_series("./observations/epi_timeseries.csv")
        print(f"   ✅ Epidemic data: {len(epi_data.times)} time points")

        fairness_data = load_fairness_series("./observations/fairness_collapse.csv")
        print(f"   ✅ Fairness data: {len(fairness_data.times)} time points")

    except Exception as e:
        print(f"   ❌ Observation adapters failed: {e}")
        return False

    # Test multiscale coupling
    print("\n2. Testing Multiscale Coupling...")
    try:
        from phase4.multiscale_coupling import (
            CellState,
            OrganismCoupler,
            PopCoupler,
        )

        # Test CellState
        cell = CellState(atp_level=0.8, damage=0.2)
        print(f"   ✅ CellState created: ATP={cell.atp_level}, Damage={cell.damage}")

        # Test OrganismCoupler
        coupler = OrganismCoupler(n_cells_per_agent=16, coupling_coeff=0.5)

        # Create mock agent
        class MockAgent:
            def __init__(self):
                self.health_score = 0.8
                self.damage = 0.2
                self.total_energy_kJ = 2000
                self.repair_rate = 0.01

        agent = MockAgent()
        cells = coupler.downscale_agent_to_cells(agent, 1.0)
        print(
            f"   ✅ Generated {len(cells)} cells with avg ATP: {np.mean([c.atp_level for c in cells]):.3f}"
        )

        # Test upscaling
        updates = coupler.upscale_cells_to_agent(cells, agent, 1.0)
        print(f"   ✅ Upscaled to agent: {len(updates)} parameters")

        # Test PopCoupler
        pop_coupler = PopCoupler(coordination_strength=0.3, fairness_weight=0.4)
        agents = [MockAgent() for _ in range(10)]

        coherence = pop_coupler.agent_to_population_coherence(agents, None)
        print(f"   ✅ Population coherence: {len(coherence)} metrics")

    except Exception as e:
        print(f"   ❌ Multiscale coupling failed: {e}")
        return False

    # Test fit metrics
    print("\n3. Testing Fit Metrics...")
    try:
        from phase4.fit_metrics import (
            compute_epidemic_fit_metrics,
            compute_fairness_fit_metrics,
            compute_km_fit_metrics,
            validate_success_criteria,
        )

        # Test KM fit metrics
        sim_times = np.array([0, 100, 200, 300, 400, 500])
        sim_survival = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
        obs_times = np.array([0, 100, 200, 300, 400, 500])
        obs_survival = np.array([1.0, 0.85, 0.72, 0.58, 0.45, 0.32])

        km_metrics = compute_km_fit_metrics(
            sim_times, sim_survival, obs_times, obs_survival
        )
        print(
            f"   ✅ KM metrics: RMSE={km_metrics['rmse']:.4f}, KS D={km_metrics['ks_statistic']:.4f}"
        )

        # Test epidemic fit metrics
        sim_cases = np.array([1, 5, 20, 50, 30, 10])
        obs_cases = np.array([1, 3, 8, 18, 35, 52])

        epi_metrics = compute_epidemic_fit_metrics(
            sim_times, sim_cases, obs_times, obs_cases
        )
        print(
            f"   ✅ Epidemic metrics: RMSE={epi_metrics['rmse']:.4f}, Peak timing error={epi_metrics['peak_timing_error']:.1f}"
        )

        # Test fairness fit metrics
        sim_fairness = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        sim_collapse = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        obs_fairness = np.array([0.85, 0.78, 0.72, 0.65, 0.58, 0.52])
        obs_collapse = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.32])

        fairness_metrics = compute_fairness_fit_metrics(
            sim_times, sim_fairness, sim_collapse, obs_times, obs_fairness, obs_collapse
        )
        print(
            f"   ✅ Fairness metrics: Phase agreement={fairness_metrics['phase_agreement']:.3f}"
        )

        # Test validation
        validation = validate_success_criteria(
            km_metrics, epi_metrics, fairness_metrics, {}
        )
        print(f"   ✅ Validation: Overall pass={validation['overall_pass']}")

    except Exception as e:
        print(f"   ❌ Fit metrics failed: {e}")
        return False

    # Test registry configuration
    print("\n4. Testing Registry Configuration...")
    try:
        import yaml

        with open("./phase4/registry.yaml") as f:
            registry_config = yaml.safe_load(f)

        print(
            f"   ✅ Registry loaded: {len(registry_config['experiment_suites'])} experiment suites"
        )
        print(f"   ✅ Datasets configured: {len(registry_config['datasets'])} types")

    except Exception as e:
        print(f"   ❌ Registry configuration failed: {e}")
        return False

    # Test Jinja template
    print("\n5. Testing Report Template...")
    try:
        from jinja2 import Environment, FileSystemLoader

        template_dir = "./phase4/templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("phase4_report.md.j2")

        # Test template rendering with dummy data
        context = {
            "timestamp": "2024-01-01 00:00:00",
            "run_id": "test_run",
            "git_commit": "abc12345",
            "config_hash": "def67890",
            "key_findings": ["Test finding 1", "Test finding 2"],
            "validation_results": {
                "km_criteria": {"overall_pass": True},
                "epi_criteria": {"overall_pass": True},
                "fairness_criteria": {"overall_pass": True},
                "unified_criteria": {"overall_pass": True},
                "ppc_criteria": {"overall_pass": True},
            },
            "energy_conservation_valid": True,
            "energy_drift": 0.005,
            "clinical_datasets": [],
            "epidemic_datasets": [],
            "fairness_datasets": [],
            "multiscale_config": {},
            "bayes_enabled": False,
            "clinical_results": [],
            "clinical_summary": {},
            "epidemic_results": [],
            "epidemic_summary": {},
            "fairness_results": [],
            "fairness_summary": {},
            "unified_results": {},
            "sensitivity_results": {},
            "ppc_results": {},
            "random_seed": 42,
            "dataset_checksums": {},
            "limitations": {"data": [], "methodological": [], "computational": []},
            "future_work": {"immediate": [], "long_term": [], "validation": []},
            "achievements": [],
            "critical_findings": [],
            "recommendations": [],
            "total_runtime": "1 hour",
            "total_simulations": 1,
        }

        report_content = template.render(**context)
        print(f"   ✅ Template rendered: {len(report_content)} characters")

    except Exception as e:
        print(f"   ❌ Report template failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL PHASE 4 MODULES TESTED SUCCESSFULLY!")
    print("=" * 60)

    return True


def test_run_phase4_imports():
    """Test that run_phase4.py can be imported without errors."""
    print("\n6. Testing run_phase4.py imports...")
    try:
        # Test imports without running main()
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_phase4", "./phase4/run_phase4.py"
        )
        run_phase4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_phase4)

        print("   ✅ run_phase4.py imports successfully")
        print("   ✅ All dependencies available")

        return True

    except Exception as e:
        print(f"   ❌ run_phase4.py import failed: {e}")
        return False


if __name__ == "__main__":
    print("Phase 4 Multi-Scale & Validation Framework - Test Suite")
    print("=" * 80)

    # Run tests
    success = test_phase4_modules()
    success &= test_run_phase4_imports()

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nPhase 4 is ready to run:")
        print("   python3 ./phase4/run_phase4.py")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
        sys.exit(1)
