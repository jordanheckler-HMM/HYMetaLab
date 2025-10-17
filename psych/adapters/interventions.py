"""
Fear-reduction and coherence enhancement interventions.

Implements various intervention strategies to reduce fear-driven aggression
and improve system coherence.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class InterventionType(Enum):
    """Types of fear-reduction interventions."""

    SUPPORT_INJECTION = "support_injection"
    COHERENCE_TRAINING = "coherence_training"
    FEAR_DESCALATION = "fear_descalation"
    SOCIAL_CONNECTION = "social_connection"
    RESOURCE_REDISTRIBUTION = "resource_redistribution"


@dataclass
class InterventionParams:
    """Parameters for intervention effectiveness."""

    # Support injection parameters
    support_boost: float = 0.2  # Increase in support_t
    support_duration: int = 10  # Steps of support injection

    # Coherence training parameters
    coherence_boost: float = 0.1  # Gradual increase in CCI
    coherence_duration: int = 20  # Steps of training

    # Fear de-escalation parameters
    fear_reduction_factor: float = 0.8  # Reduce fear persistence
    shock_dampening: float = 0.5  # Reduce shock sensitivity

    # Social connection parameters
    social_boost: float = 0.3  # Increase in social ties
    network_expansion: float = 0.2  # Increase network connectivity

    # Resource redistribution parameters
    inequality_reduction: float = 0.1  # Reduce Gini coefficient
    resource_boost: float = 0.15  # Increase resource availability

    # Cost parameters
    support_cost: float = 1.0
    coherence_cost: float = 2.0
    descalation_cost: float = 1.5
    social_cost: float = 0.8
    resource_cost: float = 3.0


class FearIntervention:
    """
    Fear-reduction intervention system.

    Implements various intervention strategies to reduce fear-driven aggression
    and improve system stability.
    """

    def __init__(
        self,
        intervention_type: InterventionType = InterventionType.SUPPORT_INJECTION,
        params: InterventionParams | None = None,
        active: bool = False,
    ):
        self.intervention_type = intervention_type
        self.params = params or InterventionParams()
        self.active = active

        # Intervention state tracking
        self.step_count = 0
        self.total_cost = 0.0
        self.effectiveness_history = []

    def apply_intervention(
        self, agent_state: dict[str, Any], group_state: dict[str, Any], step: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Apply intervention effects to agent and group states.

        Args:
            agent_state: Individual agent state
            group_state: Group-level state
            step: Current simulation step

        Returns:
            Tuple of (modified_agent_state, modified_group_state)
        """
        if not self.active:
            return agent_state, group_state

        self.step_count += 1

        # Apply intervention based on type
        if self.intervention_type == InterventionType.SUPPORT_INJECTION:
            agent_state, group_state = self._apply_support_injection(
                agent_state, group_state
            )
        elif self.intervention_type == InterventionType.COHERENCE_TRAINING:
            agent_state, group_state = self._apply_coherence_training(
                agent_state, group_state
            )
        elif self.intervention_type == InterventionType.FEAR_DESCALATION:
            agent_state, group_state = self._apply_fear_descalation(
                agent_state, group_state
            )
        elif self.intervention_type == InterventionType.SOCIAL_CONNECTION:
            agent_state, group_state = self._apply_social_connection(
                agent_state, group_state
            )
        elif self.intervention_type == InterventionType.RESOURCE_REDISTRIBUTION:
            agent_state, group_state = self._apply_resource_redistribution(
                agent_state, group_state
            )

        # Track cost
        self.total_cost += self._compute_step_cost()

        return agent_state, group_state

    def _apply_support_injection(
        self, agent_state: dict[str, Any], group_state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply support injection intervention."""

        # Increase support level
        current_support = group_state.get("support_level", 0.0)
        new_support = min(1.0, current_support + self.params.support_boost)
        group_state["support_level"] = new_support

        # Add intervention support to agent
        intervention_support = group_state.get("intervention_support", 0.0)
        group_state["intervention_support"] = (
            intervention_support + self.params.support_boost
        )

        # Reduce fear directly (more aggressive reduction)
        current_fear = agent_state.get("fear", 0.0)
        fear_reduction = self.params.support_boost * 1.0  # Strong reduction
        new_fear = max(0.0, current_fear - fear_reduction)
        agent_state["fear"] = new_fear

        return agent_state, group_state

    def _apply_coherence_training(
        self, agent_state: dict[str, Any], group_state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply coherence training intervention."""

        # Gradually increase CCI
        current_cci = agent_state.get("cci", 0.5)
        cci_increase = self.params.coherence_boost / self.params.coherence_duration
        new_cci = min(1.0, current_cci + cci_increase)
        agent_state["cci"] = new_cci

        # Update calibration accuracy as proxy for CCI
        agent_state["calibration_accuracy"] = new_cci

        # Reduce aggression probability through higher coherence
        current_fear = agent_state.get("fear", 0.0)
        from .fear_core import aggression_prob

        new_aggression_prob = aggression_prob(current_fear, new_cci)
        agent_state["aggression_prob"] = new_aggression_prob

        return agent_state, group_state

    def _apply_fear_descalation(
        self, agent_state: dict[str, Any], group_state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply fear de-escalation intervention."""

        # Reduce fear persistence
        current_fear = agent_state.get("fear", 0.0)
        fear_reduction = current_fear * (1.0 - self.params.fear_reduction_factor)
        new_fear = max(0.0, current_fear - fear_reduction)
        agent_state["fear"] = new_fear

        # Dampen shock sensitivity
        group_state["shock_dampening"] = self.params.shock_dampening

        # Update aggression probability
        current_cci = agent_state.get("cci", 0.5)
        from .fear_core import aggression_prob

        new_aggression_prob = aggression_prob(new_fear, current_cci)
        agent_state["aggression_prob"] = new_aggression_prob

        return agent_state, group_state

    def _apply_social_connection(
        self, agent_state: dict[str, Any], group_state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply social connection intervention."""

        # Increase social ties
        current_social = group_state.get("social_ties", 0.0)
        new_social = min(1.0, current_social + self.params.social_boost)
        group_state["social_ties"] = new_social

        # Expand network connectivity
        network_expansion = group_state.get("network_expansion", 0.0)
        group_state["network_expansion"] = (
            network_expansion + self.params.network_expansion
        )

        # Social connection reduces fear
        current_fear = agent_state.get("fear", 0.0)
        fear_reduction = self.params.social_boost * 0.3
        new_fear = max(0.0, current_fear - fear_reduction)
        agent_state["fear"] = new_fear

        return agent_state, group_state

    def _apply_resource_redistribution(
        self, agent_state: dict[str, Any], group_state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply resource redistribution intervention."""

        # Reduce inequality
        current_gini = group_state.get("gini", 0.0)
        new_gini = max(0.0, current_gini - self.params.inequality_reduction)
        group_state["gini"] = new_gini

        # Increase resource availability
        current_resources = group_state.get("resource_availability", 0.5)
        new_resources = min(1.0, current_resources + self.params.resource_boost)
        group_state["resource_availability"] = new_resources

        # Redistribute to agent
        current_wealth = agent_state.get("wealth", 0.5)
        wealth_boost = self.params.resource_boost * 0.2
        new_wealth = min(1.0, current_wealth + wealth_boost)
        agent_state["wealth"] = new_wealth

        # Reduced inequality reduces fear
        current_fear = agent_state.get("fear", 0.0)
        fear_reduction = self.params.inequality_reduction * 0.5
        new_fear = max(0.0, current_fear - fear_reduction)
        agent_state["fear"] = new_fear

        return agent_state, group_state

    def _compute_step_cost(self) -> float:
        """Compute cost for current intervention step."""

        if self.intervention_type == InterventionType.SUPPORT_INJECTION:
            return self.params.support_cost
        elif self.intervention_type == InterventionType.COHERENCE_TRAINING:
            return self.params.coherence_cost
        elif self.intervention_type == InterventionType.FEAR_DESCALATION:
            return self.params.descalation_cost
        elif self.intervention_type == InterventionType.SOCIAL_CONNECTION:
            return self.params.social_cost
        elif self.intervention_type == InterventionType.RESOURCE_REDISTRIBUTION:
            return self.params.resource_cost
        else:
            return 0.0

    def compute_effectiveness(
        self,
        pre_intervention_metrics: dict[str, Any],
        post_intervention_metrics: dict[str, Any],
    ) -> dict[str, float]:
        """Compute intervention effectiveness."""

        # Extract metrics
        pre_fear = pre_intervention_metrics.get("mean_fear", 0.0)
        post_fear = post_intervention_metrics.get("mean_fear", 0.0)

        pre_aggression = pre_intervention_metrics.get("aggression_rate", 0.0)
        post_aggression = post_intervention_metrics.get("aggression_rate", 0.0)

        pre_cci = pre_intervention_metrics.get("cci", 0.5)
        post_cci = post_intervention_metrics.get("cci", 0.5)

        # Compute effectiveness metrics
        fear_reduction = pre_fear - post_fear
        aggression_reduction = pre_aggression - post_aggression
        cci_improvement = post_cci - pre_cci

        # Overall effectiveness (weighted combination)
        overall_effectiveness = (
            0.4 * fear_reduction + 0.4 * aggression_reduction + 0.2 * cci_improvement
        )

        # Cost-effectiveness
        cost_effectiveness = (
            overall_effectiveness / self.total_cost if self.total_cost > 0 else 0.0
        )

        return {
            "fear_reduction": fear_reduction,
            "aggression_reduction": aggression_reduction,
            "cci_improvement": cci_improvement,
            "overall_effectiveness": overall_effectiveness,
            "cost_effectiveness": cost_effectiveness,
            "total_cost": self.total_cost,
            "intervention_type": self.intervention_type.value,
        }

    def is_complete(self) -> bool:
        """Check if intervention is complete."""

        if self.intervention_type == InterventionType.SUPPORT_INJECTION:
            return self.step_count >= self.params.support_duration
        elif self.intervention_type == InterventionType.COHERENCE_TRAINING:
            return self.step_count >= self.params.coherence_duration
        else:
            return False  # Other interventions are continuous


class InterventionManager:
    """
    Manages multiple interventions and their coordination.
    """

    def __init__(self):
        self.active_interventions: list[FearIntervention] = []
        self.intervention_history: list[dict[str, Any]] = []

    def add_intervention(
        self,
        intervention_type: InterventionType,
        params: InterventionParams | None = None,
        start_step: int = 0,
    ) -> FearIntervention:
        """Add a new intervention."""

        intervention = FearIntervention(
            intervention_type=intervention_type, params=params, active=True
        )

        self.active_interventions.append(intervention)
        return intervention

    def apply_all_interventions(
        self, agent_states: list[dict[str, Any]], group_state: dict[str, Any], step: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply all active interventions."""

        modified_agent_states = agent_states.copy()
        modified_group_state = group_state.copy()

        # Apply each intervention
        for intervention in self.active_interventions:
            if intervention.active:
                # Apply to each agent
                for i, agent_state in enumerate(modified_agent_states):
                    modified_agent_states[i], modified_group_state = (
                        intervention.apply_intervention(
                            agent_state, modified_group_state, step
                        )
                    )

                # Check if intervention is complete
                if intervention.is_complete():
                    intervention.active = False

        # Remove completed interventions
        self.active_interventions = [i for i in self.active_interventions if i.active]

        return modified_agent_states, modified_group_state

    def get_intervention_summary(self) -> dict[str, Any]:
        """Get summary of all interventions."""

        total_cost = sum(i.total_cost for i in self.active_interventions)
        active_count = len(self.active_interventions)

        intervention_types = [
            i.intervention_type.value for i in self.active_interventions
        ]

        return {
            "active_interventions": active_count,
            "total_cost": total_cost,
            "intervention_types": intervention_types,
            "history_length": len(self.intervention_history),
        }
