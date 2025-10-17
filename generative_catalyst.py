#!/usr/bin/env python3
"""
OriginChain v5 - Generative Catalyst
Automated hypothesis generation for emergence research

v5 Catalyst: Generate research hypotheses at ≥1/24h
"""
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# v5 Catalyst: Deterministic seed
CATALYST_SEED = int(os.getenv("CATALYST_SEED", "42"))
random.seed(CATALYST_SEED)


@dataclass
class Hypothesis:
    """Research hypothesis"""

    id: str
    title: str
    description: str
    domain: str
    emergence_potential: float  # 0-1
    complexity_level: str  # 'low', 'medium', 'high'
    testable: bool
    priority: int  # 1-5
    generated_at: str
    tags: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class GenerativeCatalyst:
    """
    Generative hypothesis catalyst for emergence research

    Generates research hypotheses by:
    - Combining concepts from different domains
    - Identifying emergence patterns
    - Proposing testable predictions
    - Prioritizing by potential impact
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize GenerativeCatalyst

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else CATALYST_SEED
        random.seed(self.seed)

        # Knowledge base: domains and concepts
        self.domains = {
            "consciousness": [
                "awareness",
                "experience",
                "qualia",
                "self-model",
                "metacognition",
            ],
            "complexity": [
                "emergence",
                "self-organization",
                "feedback",
                "networks",
                "criticality",
            ],
            "cognition": ["reasoning", "learning", "memory", "attention", "language"],
            "evolution": [
                "adaptation",
                "selection",
                "fitness",
                "population",
                "mutation",
            ],
            "systems": [
                "interaction",
                "boundary",
                "hierarchy",
                "dynamics",
                "stability",
            ],
            "information": [
                "entropy",
                "compression",
                "transmission",
                "integration",
                "processing",
            ],
            "meaning": ["semantics", "context", "relevance", "significance", "value"],
            "physics": ["energy", "entropy", "time", "causality", "phase-transition"],
        }

        # Hypothesis templates
        self.templates = [
            "What if {concept1} emerges from {concept2} under conditions of {concept3}?",
            "How might {concept1} and {concept2} interact to produce {concept3} in {domain} systems?",
            "Could {concept1} serve as a bridge between {concept2} and {concept3}?",
            "What role does {concept1} play in the emergence of {concept2}?",
            "Is {concept1} necessary for {concept2}, or merely sufficient?",
            "How does {concept1} scale from {domain1} to {domain2} contexts?",
            "What feedback loops connect {concept1}, {concept2}, and {concept3}?",
            "Can we measure {concept1} through its effects on {concept2}?",
            "Does {concept1} require {concept2} to manifest {concept3}?",
            "What if {concept1} is an emergent property of {concept2} dynamics?",
        ]

        # Testability patterns
        self.testable_patterns = [
            "measurable",
            "observable",
            "quantifiable",
            "computable",
            "simulatable",
            "verifiable",
            "falsifiable",
        ]

        # Generated hypotheses queue
        self.queue: list[Hypothesis] = []
        self.generation_history: list[dict] = []

    def _select_concepts(self, n: int = 3, seed_offset: int = 0) -> list[tuple]:
        """Select random concepts from domains"""
        # Re-seed for determinism
        random.seed(self.seed + seed_offset)

        concepts = []

        for _ in range(n):
            domain = random.choice(list(self.domains.keys()))
            concept = random.choice(self.domains[domain])
            concepts.append((domain, concept))

        return concepts

    def _compute_emergence_potential(self, concept_domains: list[str]) -> float:
        """
        Compute emergence potential based on domain diversity

        More diverse domains → higher emergence potential
        """
        unique_domains = len(set(concept_domains))
        total_domains = len(concept_domains)

        # Diversity bonus
        diversity = unique_domains / total_domains

        # Cross-domain bonus
        cross_domain_bonus = 0.2 if unique_domains > 1 else 0.0

        emergence_potential = min(
            1.0, diversity + cross_domain_bonus + random.uniform(0, 0.2)
        )

        return emergence_potential

    def _assess_complexity(self, description: str) -> str:
        """Assess complexity level based on description"""
        # Simple heuristic: more words → more complex
        word_count = len(description.split())

        if word_count < 20:
            return "low"
        elif word_count < 40:
            return "medium"
        else:
            return "high"

    def _is_testable(self, description: str) -> bool:
        """Check if hypothesis contains testable language"""
        description_lower = description.lower()

        for pattern in self.testable_patterns:
            if pattern in description_lower:
                return True

        # Additional checks
        testable_keywords = [
            "measure",
            "observe",
            "compute",
            "simulate",
            "test",
            "verify",
        ]
        return any(keyword in description_lower for keyword in testable_keywords)

    def _compute_priority(
        self, emergence_potential: float, testable: bool, complexity: str
    ) -> int:
        """
        Compute priority (1-5, higher is more important)

        High emergence + testable + manageable complexity → high priority
        """
        priority = 1

        # Emergence bonus
        if emergence_potential > 0.7:
            priority += 2
        elif emergence_potential > 0.5:
            priority += 1

        # Testability bonus
        if testable:
            priority += 1

        # Complexity adjustment (medium is ideal)
        if complexity == "medium":
            priority += 1
        elif complexity == "high":
            priority -= 1  # Too complex

        return max(1, min(5, priority))

    def _generate_tags(self, domains: list[str], description: str) -> list[str]:
        """Generate tags for hypothesis"""
        tags = list(set(domains))  # Domain tags

        # Content-based tags
        if "emerge" in description.lower():
            tags.append("emergence")
        if "feedback" in description.lower():
            tags.append("feedback-loops")
        if "scale" in description.lower():
            tags.append("scaling")
        if "measure" in description.lower() or "quantif" in description.lower():
            tags.append("quantitative")

        return tags

    def generate_hypothesis(self, seed_offset: int = 0) -> Hypothesis:
        """
        Generate a single research hypothesis

        Args:
            seed_offset: Offset for seed (for generating different hypotheses)

        Returns:
            Hypothesis object
        """
        # Re-seed for determinism with offset
        random.seed(self.seed + seed_offset)

        # Select concepts and domains
        concept_tuples = self._select_concepts(n=3, seed_offset=seed_offset)
        domains = [t[0] for t in concept_tuples]
        concepts = [t[1] for t in concept_tuples]

        # Select template
        template = random.choice(self.templates)

        # Fill template
        if "{concept1}" in template:
            description = template.format(
                concept1=concepts[0],
                concept2=concepts[1] if len(concepts) > 1 else "structure",
                concept3=concepts[2] if len(concepts) > 2 else "order",
                domain=random.choice(domains),
                domain1=domains[0],
                domain2=domains[1] if len(domains) > 1 else domains[0],
            )
        else:
            description = template

        # Compute attributes
        emergence_potential = self._compute_emergence_potential(domains)
        complexity = self._assess_complexity(description)
        testable = self._is_testable(description)
        priority = self._compute_priority(emergence_potential, testable, complexity)
        tags = self._generate_tags(domains, description)

        # Create title (first 60 chars of description)
        title = description[:60] + "..." if len(description) > 60 else description

        # Generate ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hypothesis_id = f"H{timestamp}_{random.randint(1000, 9999)}"

        # Create hypothesis
        hypothesis = Hypothesis(
            id=hypothesis_id,
            title=title,
            description=description,
            domain=domains[0],
            emergence_potential=emergence_potential,
            complexity_level=complexity,
            testable=testable,
            priority=priority,
            generated_at=datetime.now().isoformat(),
            tags=tags,
        )

        return hypothesis

    def generate_batch(self, n: int = 5) -> list[Hypothesis]:
        """
        Generate batch of hypotheses

        Args:
            n: Number of hypotheses to generate

        Returns:
            List of Hypothesis objects
        """
        hypotheses = []

        for i in range(n):
            hypothesis = self.generate_hypothesis(seed_offset=i)
            hypotheses.append(hypothesis)
            self.queue.append(hypothesis)

        # Record generation event
        self.generation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "count": n,
                "hypotheses": [h.id for h in hypotheses],
            }
        )

        return hypotheses

    def get_priority_queue(self, min_priority: int = 3) -> list[Hypothesis]:
        """Get hypotheses above priority threshold"""
        return [h for h in self.queue if h.priority >= min_priority]

    def get_testable_queue(self) -> list[Hypothesis]:
        """Get testable hypotheses"""
        return [h for h in self.queue if h.testable]

    def save_queue(self, output_path: Path, format: str = "markdown"):
        """
        Save hypothesis queue to file

        Args:
            output_path: Output file path
            format: 'markdown' or 'json'
        """
        if format == "markdown":
            self._save_markdown(output_path)
        elif format == "json":
            self._save_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_markdown(self, output_path: Path):
        """Save queue as markdown"""
        with open(output_path, "w") as f:
            f.write("# Genesis Queue - Research Hypotheses\n\n")
            f.write(
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(f"**Total Hypotheses**: {len(self.queue)}\n\n")

            # Stats
            testable_count = len(self.get_testable_queue())
            high_priority = len([h for h in self.queue if h.priority >= 4])

            f.write("## Summary Statistics\n\n")
            f.write(f"- **Testable**: {testable_count}/{len(self.queue)}\n")
            f.write(f"- **High Priority (≥4)**: {high_priority}\n")
            f.write(
                f"- **Mean Emergence Potential**: {sum(h.emergence_potential for h in self.queue) / len(self.queue):.2f}\n\n"
            )

            # Sort by priority
            sorted_queue = sorted(self.queue, key=lambda h: h.priority, reverse=True)

            f.write("---\n\n")
            f.write("## Hypotheses (by priority)\n\n")

            for i, h in enumerate(sorted_queue, 1):
                f.write(f"### {i}. {h.title}\n\n")
                f.write(f"**ID**: `{h.id}`  \n")
                f.write(f"**Priority**: {'⭐' * h.priority} ({h.priority}/5)  \n")
                f.write(f"**Domain**: {h.domain}  \n")
                f.write(f"**Emergence Potential**: {h.emergence_potential:.2f}  \n")
                f.write(f"**Complexity**: {h.complexity_level}  \n")
                f.write(f"**Testable**: {'✅ Yes' if h.testable else '❌ No'}  \n")
                f.write(f"**Tags**: {', '.join(h.tags)}  \n")
                f.write(f"**Generated**: {h.generated_at}  \n\n")
                f.write(f"**Hypothesis**:  \n{h.description}\n\n")
                f.write("---\n\n")

    def _save_json(self, output_path: Path):
        """Save queue as JSON"""
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_hypotheses": len(self.queue),
            "generation_history": self.generation_history,
            "hypotheses": [h.to_dict() for h in self.queue],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def simulate_daily_generation(self, days: int = 7) -> dict:
        """
        Simulate daily hypothesis generation over multiple days

        Args:
            days: Number of days to simulate

        Returns:
            Simulation statistics
        """
        daily_counts = []
        start_date = datetime.now()

        for day in range(days):
            # Generate 1-3 hypotheses per day
            daily_count = random.randint(1, 3)
            self.generate_batch(n=daily_count)
            daily_counts.append(daily_count)

        return {
            "days_simulated": days,
            "total_hypotheses": sum(daily_counts),
            "mean_per_day": sum(daily_counts) / days,
            "min_per_day": min(daily_counts),
            "max_per_day": max(daily_counts),
            "meets_target": all(c >= 1 for c in daily_counts),
        }


def main():
    """CLI for testing generative_catalyst"""
    import argparse

    parser = argparse.ArgumentParser(description="OriginChain v5 Generative Catalyst")
    parser.add_argument(
        "command", choices=["generate", "simulate", "test"], help="Command to execute"
    )
    parser.add_argument(
        "--count", type=int, default=5, help="Number of hypotheses to generate"
    )
    parser.add_argument("--output", default="genesis_queue.md", help="Output file path")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )

    args = parser.parse_args()

    if args.command == "generate":
        print(f"🌟 Generating {args.count} hypotheses...")

        catalyst = GenerativeCatalyst(seed=42)
        hypotheses = catalyst.generate_batch(n=args.count)

        print(f"\n✅ Generated {len(hypotheses)} hypotheses:")
        for i, h in enumerate(hypotheses, 1):
            print(f"\n{i}. {h.title}")
            print(f"   Priority: {'⭐' * h.priority}")
            print(f"   Testable: {'✅' if h.testable else '❌'}")
            print(f"   Emergence: {h.emergence_potential:.2f}")

        # Save
        catalyst.save_queue(Path(args.output), format=args.format)
        print(f"\n✅ Saved to: {args.output}")

    elif args.command == "simulate":
        print("🌟 Simulating 7-day hypothesis generation...")

        catalyst = GenerativeCatalyst(seed=42)
        stats = catalyst.simulate_daily_generation(days=7)

        print("\n✅ Simulation Results:")
        print(f"   Days: {stats['days_simulated']}")
        print(f"   Total hypotheses: {stats['total_hypotheses']}")
        print(f"   Mean per day: {stats['mean_per_day']:.1f}")
        print(f"   Range: {stats['min_per_day']}-{stats['max_per_day']}")
        print(f"   Meets ≥1/day target: {'✅' if stats['meets_target'] else '❌'}")

        # Save
        catalyst.save_queue(Path(args.output), format=args.format)
        print(f"\n✅ Saved {len(catalyst.queue)} hypotheses to: {args.output}")

    elif args.command == "test":
        print("🌟 Testing Generative Catalyst...")

        catalyst = GenerativeCatalyst(seed=42)

        # Test single generation
        h = catalyst.generate_hypothesis()
        print("\n✅ Generated hypothesis:")
        print(f"   Title: {h.title}")
        print(f"   Domain: {h.domain}")
        print(f"   Priority: {h.priority}/5")
        print(f"   Testable: {h.testable}")
        print(f"   Emergence: {h.emergence_potential:.2f}")

        # Test batch
        batch = catalyst.generate_batch(n=3)
        print(f"\n✅ Generated batch of {len(batch)} hypotheses")

        # Test daily simulation
        stats = catalyst.simulate_daily_generation(days=1)
        print(f"\n✅ Daily simulation: {stats['total_hypotheses']} hypotheses")
        print(f"   Meets ≥1/day: {'✅' if stats['meets_target'] else '❌'}")


if __name__ == "__main__":
    main()
