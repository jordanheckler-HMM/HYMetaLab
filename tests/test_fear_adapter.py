"""
Tests for fear-violence adaptation framework.

Ensures fear dynamics work correctly without breaking existing simulations.
"""

import numpy as np

from psych.adapters.fear_core import FearParams, aggression_prob, update_fear
from psych.adapters.fear_hooks import FearChannel
from psych.adapters.interventions import FearIntervention, InterventionType


def test_fear_update_monotonicity():
    """Test that fear increases with shock and inequality."""

    params = FearParams()

    # Test shock monotonicity
    fear_low_shock = update_fear(
        current_fear=0.3,
        shock_level=0.2,
        inequality=0.0,
        coherence=0.5,
        group_state={},
        params=params,
    )

    fear_high_shock = update_fear(
        current_fear=0.3,
        shock_level=0.8,
        inequality=0.0,
        coherence=0.5,
        group_state={},
        params=params,
    )

    assert fear_high_shock > fear_low_shock, "Higher shock should increase fear"

    # Test inequality monotonicity
    fear_low_inequality = update_fear(
        current_fear=0.3,
        shock_level=0.0,
        inequality=0.2,
        coherence=0.5,
        group_state={},
        params=params,
    )

    fear_high_inequality = update_fear(
        current_fear=0.3,
        shock_level=0.0,
        inequality=0.8,
        coherence=0.5,
        group_state={},
        params=params,
    )

    assert (
        fear_high_inequality > fear_low_inequality
    ), "Higher inequality should increase fear"


def test_cci_moderation():
    """Test that higher CCI reduces aggression probability."""

    params = FearParams()

    # Fixed fear level, different CCI levels
    fear_level = 0.7  # High fear

    aggression_low_cci = aggression_prob(fear=fear_level, coherence=0.2, params=params)
    aggression_high_cci = aggression_prob(fear=fear_level, coherence=0.8, params=params)

    assert (
        aggression_high_cci < aggression_low_cci
    ), "Higher CCI should reduce aggression probability"


def test_fear_channel_disabled():
    """Test that disabled fear channel doesn't modify agent state."""

    # Disabled channel
    channel_disabled = FearChannel(enable=False)

    # Enabled channel
    channel_enabled = FearChannel(enable=True)

    initial_state = {"energy": 0.8, "survival_prob": 0.9, "rand_aggr": 0.5}

    group_state = {"shock_level": 0.5, "gini": 0.3}

    # Test disabled channel
    result_disabled = channel_disabled.before_commit(
        t_now=0, agent_state=initial_state.copy(), group_state=group_state
    )

    # Should return unchanged state
    assert result_disabled == initial_state, "Disabled channel should not modify state"

    # Test enabled channel
    result_enabled = channel_enabled.before_commit(
        t_now=0, agent_state=initial_state.copy(), group_state=group_state
    )

    # Should add fear and aggression fields
    assert "fear" in result_enabled, "Enabled channel should add fear field"
    assert (
        "aggression_event" in result_enabled
    ), "Enabled channel should add aggression_event field"
    assert (
        "aggression_prob" in result_enabled
    ), "Enabled channel should add aggression_prob field"


def test_intervention_support():
    """Test that support intervention reduces fear."""

    intervention = FearIntervention(
        intervention_type=InterventionType.SUPPORT_INJECTION, active=True
    )

    # Test with a higher initial fear to ensure reduction is visible
    initial_agent_state = {"fear": 0.8}
    initial_group_state = {"support_level": 0.0}

    # Store original values before intervention
    original_fear = initial_agent_state["fear"]
    original_support = initial_group_state["support_level"]

    # Apply intervention
    modified_agent, modified_group = intervention.apply_intervention(
        initial_agent_state, initial_group_state, 0
    )

    # Fear should be reduced
    assert (
        modified_agent["fear"] < original_fear
    ), f"Support intervention should reduce fear: {modified_agent['fear']} < {original_fear}"

    # Support level should be increased
    assert (
        modified_group["support_level"] > original_support
    ), f"Support intervention should increase support level: {modified_group['support_level']} > {original_support}"


def test_intervention_coherence():
    """Test that coherence training increases CCI."""

    intervention = FearIntervention(
        intervention_type=InterventionType.COHERENCE_TRAINING, active=True
    )

    initial_agent_state = {"cci": 0.5}
    initial_group_state = {}

    # Store original CCI before intervention
    original_cci = initial_agent_state["cci"]

    # Apply intervention
    modified_agent, modified_group = intervention.apply_intervention(
        initial_agent_state, initial_group_state, 0
    )

    # CCI should be increased (even if only slightly)
    expected_increase = (
        intervention.params.coherence_boost / intervention.params.coherence_duration
    )
    assert (
        modified_agent["cci"] >= original_cci + expected_increase * 0.9
    ), f"Coherence training should increase CCI: {modified_agent['cci']} >= {original_cci + expected_increase * 0.9}"


def test_fear_contagion():
    """Test fear contagion effects."""

    from psych.adapters.fear_core import compute_fear_contagion

    # Create network and fear levels
    n_agents = 5
    agent_fears = np.array([0.2, 0.8, 0.3, 0.4, 0.1])  # Agent 1 has high fear

    # Simple chain network
    network = np.zeros((n_agents, n_agents))
    for i in range(n_agents - 1):
        network[i, i + 1] = 1
        network[i + 1, i] = 1

    # Compute contagion
    contagion_effects = compute_fear_contagion(agent_fears, network, eta=0.1)

    # Agent 0 (next to high-fear agent 1) should have positive contagion
    assert (
        contagion_effects[0] > 0
    ), "Agent next to high-fear agent should have positive contagion"

    # Agent 2 (next to high-fear agent 1) should have positive contagion
    assert (
        contagion_effects[2] > 0
    ), "Agent next to high-fear agent should have positive contagion"


def test_aggression_probability_bounds():
    """Test that aggression probability stays within bounds."""

    params = FearParams()

    # Test extreme values
    min_aggression = aggression_prob(fear=0.0, coherence=1.0, params=params)
    max_aggression = aggression_prob(fear=1.0, coherence=0.0, params=params)

    assert (
        0.0 <= min_aggression <= 1.0
    ), "Aggression probability should be bounded [0,1]"
    assert (
        0.0 <= max_aggression <= 1.0
    ), "Aggression probability should be bounded [0,1]"

    # Minimum should be very low
    assert min_aggression < 0.1, "Min aggression should be very low"

    # Maximum should be high but not 1.0 (due to CCI moderation)
    assert max_aggression > 0.5, "Max aggression should be high"


def test_fear_bounds():
    """Test that fear stays within bounds."""

    params = FearParams()

    # Test extreme inputs
    min_fear = update_fear(
        current_fear=0.0,
        shock_level=0.0,
        inequality=0.0,
        coherence=1.0,
        group_state={"support_level": 1.0},
        params=params,
    )

    max_fear = update_fear(
        current_fear=1.0,
        shock_level=1.0,
        inequality=1.0,
        coherence=0.0,
        group_state={"support_level": 0.0},
        params=params,
    )

    assert 0.0 <= min_fear <= 1.0, "Fear should be bounded [0,1]"
    assert 0.0 <= max_fear <= 1.0, "Fear should be bounded [0,1]"


def test_fear_channel_statistics():
    """Test fear channel statistics computation."""

    channel = FearChannel(enable=True, track_history=True)

    # Simulate some steps
    for i in range(10):
        channel.before_commit(
            t_now=i,
            agent_state={"energy": 0.8, "cci": 0.6},
            group_state={"shock_level": 0.5, "gini": 0.3},
        )

    stats = channel.get_statistics()

    # Check that statistics are computed
    assert "mean_fear" in stats, "Statistics should include mean_fear"
    assert (
        "mean_aggression_rate" in stats
    ), "Statistics should include mean_aggression_rate"
    assert "total_steps" in stats, "Statistics should include total_steps"

    assert stats["total_steps"] == 10, "Should track correct number of steps"
    assert 0.0 <= stats["mean_fear"] <= 1.0, "Mean fear should be bounded"


def test_intervention_effectiveness():
    """Test intervention effectiveness computation."""

    intervention = FearIntervention(
        intervention_type=InterventionType.SUPPORT_INJECTION, active=True
    )

    # Simulate some cost
    intervention.total_cost = 5.0

    pre_metrics = {"mean_fear": 0.6, "aggression_rate": 0.3, "cci": 0.5}

    post_metrics = {"mean_fear": 0.4, "aggression_rate": 0.2, "cci": 0.5}

    effectiveness = intervention.compute_effectiveness(pre_metrics, post_metrics)

    assert effectiveness["fear_reduction"] > 0, "Should show fear reduction"
    assert effectiveness["aggression_reduction"] > 0, "Should show aggression reduction"
    assert (
        effectiveness["cost_effectiveness"] > 0
    ), "Should show positive cost-effectiveness"


if __name__ == "__main__":
    # Run tests
    test_fear_update_monotonicity()
    test_cci_moderation()
    test_fear_channel_disabled()
    test_intervention_support()
    test_intervention_coherence()
    test_fear_contagion()
    test_aggression_probability_bounds()
    test_fear_bounds()
    test_fear_channel_statistics()
    test_intervention_effectiveness()

    print("All fear-violence adapter tests passed!")
