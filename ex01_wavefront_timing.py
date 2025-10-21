#!/usr/bin/env python3
"""
EX01 - Finite-Speed Wavefront Timing
HYMetaLab Lab Tech Implementation

Measure propagation speed of light-like waves in vacuum using FDTD simulation.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from light_experiments_framework import LightExperiment, ExperimentConfig, find_arrival_time, compute_speed_estimate
import csv
from pathlib import Path


class EX01_WavefrontTiming(LightExperiment):
    """EX01: Finite-Speed Wavefront Timing experiment"""
    
    def setup(self):
        """Setup wavefront timing experiment"""
        # Place 10 evenly spaced detectors
        detector_spacing = self.config.grid_size // 12  # leave room at edges
        start_pos = detector_spacing * 2
        
        self.detector_positions = []
        for i in range(10):
            pos = start_pos + i * detector_spacing
            self.detector_positions.append(pos)
            self.solver.add_detector(pos)
        
        # Source parameters for Gaussian pulse
        self.pulse_width = 10 * self.config.dt  # pulse duration
        self.pulse_center = 50 * self.config.dt  # pulse peak time
        
        # Initialize tracking arrays
        self.time_series = []  # record field at each detector over time
        self.detector_signals = [[] for _ in range(len(self.detector_positions))]
        
        print(f"Setup complete: {len(self.detector_positions)} detectors from x={start_pos*self.config.dx:.3f}m to x={self.detector_positions[-1]*self.config.dx:.3f}m")
    
    def run(self):
        """Execute wavefront timing simulation"""
        print("Running FDTD simulation...")
        
        for t_step in range(self.config.time_steps):
            current_time = t_step * self.config.dt
            
            # Generate Gaussian pulse source
            source_amplitude = self.solver.gaussian_pulse(
                current_time, self.pulse_center, self.pulse_width
            )
            
            # FDTD time stepping
            self.solver.step_Ez(source_amplitude)
            self.solver.step_Hy()
            
            # Record detector readings
            readings = self.solver.get_detector_readings()
            for i, reading in enumerate(readings):
                self.detector_signals[i].append(reading)
            
            # Store time point for analysis
            if t_step % 10 == 0:  # subsample for storage efficiency
                self.raw_data.append({
                    'time': current_time,
                    'step': t_step,
                    **{f'detector_{i}': readings[i] for i in range(len(readings))}
                })
    
    def analyze(self):
        """Analyze arrival times and compute speed estimates"""
        print("Analyzing wavefront arrival times...")
        
        arrival_times = []
        distances = []
        
        # Find arrival time at each detector
        for i, signal in enumerate(self.detector_signals):
            signal_array = np.array(signal)
            arrival_time = find_arrival_time(signal_array, self.config.dt, threshold=0.1)
            
            if arrival_time > 0:  # valid detection
                arrival_times.append(arrival_time)
                distance = self.detector_positions[i] * self.config.dx
                distances.append(distance)
        
        if len(arrival_times) < 2:
            self.results['error'] = "Insufficient valid detections"
            return
        
        # Compute speed estimates between consecutive detectors
        c_estimates = []
        for i in range(1, len(arrival_times)):
            dt = arrival_times[i] - arrival_times[i-1]
            dx = distances[i] - distances[i-1]
            if dt > 0:
                c_est = dx / dt
                c_estimates.append(c_est)
        
        # Overall speed estimate from linear fit
        c_fit, c_std = compute_speed_estimate(distances, arrival_times)
        
        # Statistics
        c_mean = np.mean(c_estimates) if c_estimates else 0.0
        c_variance = np.var(c_estimates) if len(c_estimates) > 1 else 0.0
        c_std_dev = np.std(c_estimates) if len(c_estimates) > 1 else 0.0
        
        # Percent deviation from theoretical c
        theoretical_c = self.config.c
        percent_dev = abs(c_mean - theoretical_c) / theoretical_c * 100 if c_mean > 0 else 100
        
        # Store results
        self.results.update({
            'n_detections': len(arrival_times),
            'arrival_times': arrival_times,
            'distances_m': distances,
            'c_estimates_ms': c_estimates,
            'c_mean_ms': c_mean,
            'c_std_dev_ms': c_std_dev,
            'c_variance': c_variance,
            'c_fit_ms': c_fit,
            'c_fit_std_ms': c_std,
            'theoretical_c_ms': theoretical_c,
            'percent_deviation': percent_dev,
            'variance_percent': (c_std_dev / c_mean * 100) if c_mean > 0 else 0,
            'pass_variance_check': (c_std_dev / c_mean * 100) < 2.0 if c_mean > 0 else False
        })
        
        print(f"Speed estimate: {c_mean:.3e} ± {c_std_dev:.3e} m/s")
        print(f"Theoretical c: {theoretical_c:.3e} m/s")
        print(f"Deviation: {percent_dev:.2f}%")
        print(f"Variance check (<2%): {'PASS' if self.results['pass_variance_check'] else 'FAIL'}")
        
        # Save arrival times CSV
        self._save_arrival_times_csv(distances, arrival_times)
        
        # Generate plots
        self._create_plots()
    
    def _save_arrival_times_csv(self, distances: list, arrival_times: list):
        """Save arrival times to CSV file"""
        csv_path = self.config.output_dir / "arrival_times.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['detector_idx', 'distance_m', 'arrival_time_s', 'position_idx'])
            for i, (dist, time) in enumerate(zip(distances, arrival_times)):
                writer.writerow([i, dist, time, self.detector_positions[i]])
    
    def _create_plots(self):
        """Create visualization plots"""
        # Plot 1: Wavefront propagation snapshot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Spatial field distribution at peak time
        x_positions = np.arange(self.config.grid_size) * self.config.dx
        peak_time_idx = int(self.pulse_center / self.config.dt) + 100
        if peak_time_idx < len(self.detector_signals[0]):
            field_snapshot = []
            for i in range(self.config.grid_size):
                if i in self.solver.detectors:
                    det_idx = self.solver.detectors.index(i)
                    field_snapshot.append(self.detector_signals[det_idx][peak_time_idx])
                else:
                    field_snapshot.append(0.0)  # approximate
            
            ax1.plot(x_positions, field_snapshot, 'b-', linewidth=2, label='Ez field')
            ax1.axvline(self.solver.source_pos * self.config.dx, color='red', linestyle='--', label='Source')
            
            # Mark detectors
            for pos in self.detector_positions:
                ax1.axvline(pos * self.config.dx, color='gray', alpha=0.5, linestyle=':')
            
            ax1.set_xlabel('Position (m)')
            ax1.set_ylabel('Electric Field (V/m)')
            ax1.set_title('Wavefront Snapshot')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time series at selected detectors
        time_array = np.arange(len(self.detector_signals[0])) * self.config.dt
        
        # Plot every 2nd detector to avoid clutter
        for i in range(0, len(self.detector_signals), 2):
            distance = self.detector_positions[i] * self.config.dx
            ax2.plot(time_array * 1e9, self.detector_signals[i], 
                    label=f'Det {i} (x={distance:.3f}m)', alpha=0.8)
        
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Electric Field (V/m)')
        ax2.set_title('Detector Time Series')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / "plot_wavefront.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Speed analysis
        if 'distances_m' in self.results and len(self.results['distances_m']) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            distances = self.results['distances_m']
            arrival_times = self.results['arrival_times']
            
            # Scatter plot of distance vs time
            ax.scatter(np.array(arrival_times) * 1e9, distances, 
                      color='blue', s=50, label='Detections', zorder=5)
            
            # Fit line
            if self.results['c_fit_ms'] > 0:
                t_fit = np.linspace(min(arrival_times), max(arrival_times), 100)
                d_fit = self.results['c_fit_ms'] * (t_fit - arrival_times[0]) + distances[0]
                ax.plot(t_fit * 1e9, d_fit, 'r--', linewidth=2, 
                       label=f'Fit: c = {self.results["c_fit_ms"]:.2e} m/s')
            
            ax.set_xlabel('Arrival Time (ns)')
            ax.set_ylabel('Distance (m)')
            ax.set_title('Wavefront Speed Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.config.output_dir / "plot_speed_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()


def run_ex01_parameter_sweep():
    """Run EX01 with parameter sweep as specified"""
    dx_values = [0.01, 0.02, 0.05]
    seeds = [11, 17, 23]
    
    all_results = []
    
    for dx in dx_values:
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Running EX01: dx={dx}, seed={seed}")
            print('='*50)
            
            config = ExperimentConfig(
                experiment_id="EX01_LIGHT_SPEED",
                seed=seed,
                dx=dx,
                grid_size=1000,
                time_steps=2000
            )
            
            experiment = EX01_WavefrontTiming(config)
            experiment.execute()
            
            # Store key results for summary
            result_summary = {
                'dx': dx,
                'seed': seed,
                'c_mean': experiment.results.get('c_mean_ms', 0),
                'c_std_dev': experiment.results.get('c_std_dev_ms', 0),
                'percent_deviation': experiment.results.get('percent_deviation', 100),
                'variance_percent': experiment.results.get('variance_percent', 100),
                'pass_check': experiment.results.get('pass_variance_check', False)
            }
            all_results.append(result_summary)
    
    # Save combined results
    summary_path = Path("HYMetaLab/light/EX01_LIGHT_SPEED/combined_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n{'='*50}")
    print("EX01 Parameter Sweep Complete")
    print(f"Combined results saved to: {summary_path}")
    print('='*50)
    
    # Print summary table
    print("\nSUMMARY:")
    print("dx      seed  c_mean(m/s)     std_dev     %dev    %var   PASS")
    print("-" * 65)
    for r in all_results:
        print(f"{r['dx']:<6} {r['seed']:<4} {r['c_mean']:.3e} {r['c_std_dev']:.2e} "
              f"{r['percent_deviation']:<6.2f} {r['variance_percent']:<6.2f} {r['pass_check']}")


if __name__ == "__main__":
    run_ex01_parameter_sweep()