#!/usr/bin/env python3
"""
Light Speed Series Experiment Framework
HYMetaLab Lab Tech Implementation

Core FDTD solver and experiment base classes for light propagation studies.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv


@dataclass
class ExperimentConfig:
    """Configuration for light speed experiments"""
    experiment_id: str
    seed: int
    dx: float = 0.01  # spatial step (meters)
    dt: float = None  # time step (auto-computed for CFL stability)
    grid_size: int = 1000  # number of spatial points
    time_steps: int = 2000  # simulation duration
    wavelength: float = 632.8e-9  # meters (HeNe laser)
    c: float = 2.998e8  # speed of light (m/s)
    output_dir: Path = None
    
    def __post_init__(self):
        if self.dt is None:
            # CFL condition: dt <= dx/(c*sqrt(2))
            self.dt = self.dx / (self.c * np.sqrt(2)) * 0.9  # safety factor
        if self.output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            self.output_dir = Path(f"HYMetaLab/light/{self.experiment_id}/{date_str}/seed_{self.seed:02d}")
        self.output_dir.mkdir(parents=True, exist_ok=True)


class FDTD1D:
    """1D Finite-Difference Time-Domain electromagnetic solver"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.nx = config.grid_size
        self.nt = config.time_steps
        self.dx = config.dx
        self.dt = config.dt
        self.c = config.c
        
        # Initialize fields
        self.Ez = np.zeros(self.nx)  # Electric field
        self.Hy = np.zeros(self.nx)  # Magnetic field
        
        # Material properties (can be modified for experiments)
        self.epsilon_r = np.ones(self.nx)  # relative permittivity
        self.mu_r = np.ones(self.nx)       # relative permeability
        self.sigma = np.zeros(self.nx)     # conductivity
        
        # FDTD coefficients
        self._compute_coefficients()
        
        # Source parameters
        self.source_pos = self.nx // 10  # source position
        self.detectors = []  # list of detector positions
        
    def _compute_coefficients(self):
        """Compute FDTD update coefficients"""
        eps_0 = 8.854e-12  # vacuum permittivity
        mu_0 = 4*np.pi*1e-7  # vacuum permeability
        
        self.ca = (1 - self.sigma * self.dt / (2 * eps_0 * self.epsilon_r)) / \
                  (1 + self.sigma * self.dt / (2 * eps_0 * self.epsilon_r))
        self.cb = self.dt / (eps_0 * self.epsilon_r * self.dx) / \
                  (1 + self.sigma * self.dt / (2 * eps_0 * self.epsilon_r))
        self.da = np.ones(self.nx)
        self.db = self.dt / (mu_0 * self.mu_r * self.dx)
    
    def set_material(self, start_idx: int, end_idx: int, epsilon_r: float, mu_r: float = 1.0):
        """Set material properties in a region"""
        self.epsilon_r[start_idx:end_idx] = epsilon_r
        self.mu_r[start_idx:end_idx] = mu_r
        self._compute_coefficients()
    
    def add_detector(self, position: int):
        """Add field detector at given position"""
        self.detectors.append(position)
    
    def gaussian_pulse(self, t: float, t0: float, width: float) -> float:
        """Gaussian pulse source"""
        return np.exp(-((t - t0) / width) ** 2)
    
    def sinusoidal_source(self, t: float, frequency: float) -> float:
        """Sinusoidal source"""
        return np.sin(2 * np.pi * frequency * t)
    
    def step_Ez(self, source_value: float = 0.0):
        """Update electric field"""
        # Update Ez everywhere except boundaries
        self.Ez[1:-1] = self.ca[1:-1] * self.Ez[1:-1] + \
                        self.cb[1:-1] * (self.Hy[1:-1] - self.Hy[:-2])
        
        # Apply source
        if source_value != 0.0:
            self.Ez[self.source_pos] += source_value
    
    def step_Hy(self):
        """Update magnetic field"""
        self.Hy[:-1] = self.da[:-1] * self.Hy[:-1] + \
                       self.db[:-1] * (self.Ez[1:] - self.Ez[:-1])
    
    def get_detector_readings(self) -> List[float]:
        """Get current Ez values at all detector positions"""
        return [self.Ez[pos] for pos in self.detectors]


class LightExperiment(ABC):
    """Base class for light speed experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.solver = FDTD1D(config)
        self.results = {}
        self.raw_data = []
        
        # Set random seed for reproducibility
        np.random.seed(config.seed)
    
    @abstractmethod
    def setup(self):
        """Setup experiment-specific parameters"""
        pass
    
    @abstractmethod
    def run(self):
        """Execute the experiment"""
        pass
    
    @abstractmethod
    def analyze(self):
        """Analyze results and compute metrics"""
        pass
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        else:
            return obj
    
    def save_results(self):
        """Save all experiment outputs"""
        # Save raw data
        if self.raw_data:
            csv_path = self.config.output_dir / "raw_data.csv"
            with open(csv_path, 'w', newline='') as f:
                if self.raw_data:
                    writer = csv.DictWriter(f, fieldnames=self.raw_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.raw_data)
        
        # Save summary (convert numpy types for JSON serialization)
        summary_path = self.config.output_dir / "summary.json"
        json_results = self._convert_for_json(self.results)
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save experiment notes
        notes_path = self.config.output_dir / "notes.md"
        with open(notes_path, 'w') as f:
            f.write(f"# {self.config.experiment_id}\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Seed: {self.config.seed}\n")
            f.write(f"Grid size: {self.config.grid_size}\n")
            f.write(f"dx: {self.config.dx} m\n")
            f.write(f"dt: {self.config.dt:.2e} s\n")
            f.write("\n## Results Summary\n")
            for key, value in self.results.items():
                f.write(f"- {key}: {value}\n")
    
    def execute(self):
        """Run complete experiment workflow"""
        print(f"Starting {self.config.experiment_id} with seed {self.config.seed}")
        self.setup()
        self.run()
        self.analyze()
        self.save_results()
        print(f"Completed {self.config.experiment_id}, results saved to {self.config.output_dir}")


def find_arrival_time(signal: np.ndarray, dt: float, threshold: float = 0.1) -> float:
    """Find arrival time of wavefront using threshold detection"""
    max_val = np.max(np.abs(signal))
    threshold_val = threshold * max_val
    
    for i, val in enumerate(signal):
        if np.abs(val) > threshold_val:
            return i * dt
    
    return -1  # not found


def compute_speed_estimate(distances: List[float], arrival_times: List[float]) -> Tuple[float, float]:
    """Compute speed estimate from distance/time data"""
    if len(distances) != len(arrival_times) or len(distances) < 2:
        return 0.0, 0.0
    
    # Linear fit to get speed (slope of distance vs time)
    coeffs = np.polyfit(arrival_times, distances, 1)
    speed_est = coeffs[0]
    
    # Compute residuals for uncertainty estimate
    fitted_distances = np.polyval(coeffs, arrival_times)
    residuals = np.array(distances) - fitted_distances
    std_err = np.std(residuals)
    
    return speed_est, std_err


if __name__ == "__main__":
    # Test basic FDTD setup
    config = ExperimentConfig("TEST", seed=42, grid_size=100, time_steps=50)
    solver = FDTD1D(config)
    
    # Add some detectors
    for i in range(10, 90, 10):
        solver.add_detector(i)
    
    print(f"FDTD solver initialized:")
    print(f"  Grid: {solver.nx} points, dx={solver.dx} m")
    print(f"  Time: {solver.nt} steps, dt={solver.dt:.2e} s")
    print(f"  CFL number: {solver.c * solver.dt / solver.dx:.3f}")
    print(f"  Detectors at: {solver.detectors}")