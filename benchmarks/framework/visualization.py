"""Visualization tools for benchmark results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np

from .base import BenchmarkResult


class BenchmarkVisualizer:
    """Create visualizations for benchmark results."""
    
    def __init__(self, style: str = "seaborn"):
        """Initialize visualizer with matplotlib style."""
        if style in plt.style.available:
            plt.style.use(style)
        self.figures: List[Figure] = []
    
    def plot_timing_distribution(
        self,
        result: BenchmarkResult,
        bins: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Figure:
        """Plot timing distribution histogram."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        times = [result.mean_time] * result.iterations  # Simplified
        ax1.hist(times, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(result.mean_time, color='red', linestyle='--', label=f'Mean: {result.mean_time:.4f}s')
        ax1.axvline(result.median_time, color='green', linestyle='--', label=f'Median: {result.median_time:.4f}s')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Timing Distribution - {result.name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        percentile_values = [
            result.min_time,
            result.percentiles.get(25, result.min_time),
            result.median_time,
            result.percentiles.get(75, result.max_time),
            result.max_time
        ]
        
        ax2.boxplot([percentile_values], labels=[result.name])
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Timing Statistics')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_memory_usage(
        self,
        result: BenchmarkResult,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """Plot memory usage over time."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if result.memory_samples:
            x = list(range(len(result.memory_samples)))
            y = result.memory_samples
            
            ax.plot(x, y, 'b-', alpha=0.7, linewidth=2)
            ax.fill_between(x, 0, y, alpha=0.3)
            
            # Add statistics lines
            ax.axhline(result.avg_memory_mb, color='green', linestyle='--', 
                      label=f'Average: {result.avg_memory_mb:.1f} MB')
            ax.axhline(result.peak_memory_mb, color='red', linestyle='--',
                      label=f'Peak: {result.peak_memory_mb:.1f} MB')
            
            ax.set_xlabel('Sample')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title(f'Memory Usage - {result.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No memory data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Memory Usage - {result.name}')
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_comparison(
        self,
        results: List[BenchmarkResult],
        metric: str = "mean_time",
        figsize: Tuple[int, int] = (12, 8)
    ) -> Figure:
        """Compare multiple benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Extract data
        names = [r.name for r in results]
        
        # 1. Timing comparison
        ax = axes[0]
        times = [r.mean_time for r in results]
        bars = ax.bar(names, times, color='skyblue', edgecolor='navy')
        
        # Add value labels
        for bar, time in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time:.4f}', ha='center', va='bottom')
        
        ax.set_ylabel('Mean Time (seconds)')
        ax.set_title('Execution Time Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Throughput comparison
        ax = axes[1]
        throughputs = [r.throughput or 0 for r in results]
        bars = ax.bar(names, throughputs, color='lightgreen', edgecolor='darkgreen')
        
        for bar, tp in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{tp:.1f}', ha='center', va='bottom')
        
        ax.set_ylabel('Throughput (ops/second)')
        ax.set_title('Throughput Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Memory comparison
        ax = axes[2]
        memories = [r.peak_memory_mb for r in results]
        bars = ax.bar(names, memories, color='lightcoral', edgecolor='darkred')
        
        for bar, mem in zip(bars, memories):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{mem:.1f}', ha='center', va='bottom')
        
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Variability comparison (CV)
        ax = axes[3]
        cvs = [r.std_dev / r.mean_time if r.mean_time > 0 else 0 for r in results]
        bars = ax.bar(names, cvs, color='plum', edgecolor='purple')
        
        for bar, cv in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{cv:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Performance Variability')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels if needed
        for ax in axes:
            if len(names) > 5:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Benchmark Comparison', fontsize=16)
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_trend(
        self,
        results: List[BenchmarkResult],
        metrics: List[str] = ["mean_time", "throughput", "peak_memory_mb"],
        figsize: Tuple[int, int] = (14, 10)
    ) -> Figure:
        """Plot performance trends over time."""
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        
        if len(metrics) == 1:
            axes = [axes]
        
        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)
        timestamps = [r.timestamp for r in sorted_results]
        
        for ax, metric in zip(axes, metrics):
            values = []
            for r in sorted_results:
                if metric == "mean_time":
                    values.append(r.mean_time)
                elif metric == "throughput":
                    values.append(r.throughput or 0)
                elif metric == "peak_memory_mb":
                    values.append(r.peak_memory_mb)
                else:
                    # Try to get from custom metrics
                    values.append(r.custom_metrics.get(metric, 0))
            
            ax.plot(timestamps, values, 'b-o', markersize=8, linewidth=2)
            
            # Add trend line
            if len(values) > 2:
                z = np.polyfit(range(len(values)), values, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(range(len(values))), "r--", alpha=0.8, 
                       label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format dates on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        axes[-1].set_xlabel('Timestamp')
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Performance Trends - {sorted_results[0].name}', fontsize=16)
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_latency_percentiles(
        self,
        result: BenchmarkResult,
        percentiles: List[int] = [50, 90, 95, 99],
        figsize: Tuple[int, int] = (10, 6)
    ) -> Figure:
        """Plot latency percentiles."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if result.percentiles:
            x = list(result.percentiles.keys())
            y = list(result.percentiles.values())
            
            ax.plot(x, y, 'b-o', markersize=10, linewidth=2)
            
            # Highlight specific percentiles
            for p in percentiles:
                if p in result.percentiles:
                    ax.axhline(result.percentiles[p], color='gray', 
                              linestyle='--', alpha=0.5)
                    ax.text(max(x) * 0.02, result.percentiles[p], 
                           f'p{p}: {result.percentiles[p]:.4f}s',
                           va='bottom')
            
            ax.set_xlabel('Percentile')
            ax.set_ylabel('Latency (seconds)')
            ax.set_title(f'Latency Percentiles - {result.name}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
        else:
            ax.text(0.5, 0.5, 'No percentile data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def create_dashboard(
        self,
        results: Union[BenchmarkResult, List[BenchmarkResult]],
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (16, 12)
    ) -> Figure:
        """Create a comprehensive dashboard with multiple plots."""
        if isinstance(results, BenchmarkResult):
            results = [results]
        
        fig = plt.figure(figsize=figsize)
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Latest result for detailed plots
        latest = results[-1]
        
        # 1. Timing distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        times = [latest.mean_time] * latest.iterations
        ax1.hist(times, bins=30, alpha=0.7, color='blue')
        ax1.axvline(latest.mean_time, color='red', linestyle='--')
        ax1.set_title('Timing Distribution')
        ax1.set_xlabel('Time (s)')
        
        # 2. Memory usage (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if latest.memory_samples:
            ax2.plot(latest.memory_samples, 'g-', linewidth=2)
            ax2.set_title('Memory Usage')
            ax2.set_ylabel('Memory (MB)')
        
        # 3. Key metrics (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        metrics_text = (
            f"Benchmark: {latest.name}\n"
            f"Mean Time: {latest.mean_time:.4f}s\n"
            f"Std Dev: {latest.std_dev:.4f}s\n"
            f"Throughput: {latest.throughput:.2f} ops/s\n"
            f"Peak Memory: {latest.peak_memory_mb:.1f} MB\n"
            f"Iterations: {latest.iterations}"
        )
        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Comparison chart (middle row, full width)
        if len(results) > 1:
            ax4 = fig.add_subplot(gs[1, :])
            names = [f"Run {i+1}" for i in range(len(results))]
            times = [r.mean_time for r in results]
            bars = ax4.bar(names, times, color='skyblue')
            ax4.set_ylabel('Mean Time (s)')
            ax4.set_title('Performance Across Runs')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Trend analysis (bottom left)
        if len(results) > 1:
            ax5 = fig.add_subplot(gs[2, 0])
            timestamps = [r.timestamp for r in results]
            throughputs = [r.throughput or 0 for r in results]
            ax5.plot(timestamps, throughputs, 'g-o', linewidth=2)
            ax5.set_ylabel('Throughput (ops/s)')
            ax5.set_title('Throughput Trend')
            ax5.grid(True, alpha=0.3)
            
        # 6. Percentiles (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        if latest.percentiles:
            percentiles = list(latest.percentiles.keys())
            values = [latest.percentiles[p] * 1000 for p in percentiles]  # Convert to ms
            ax6.plot(percentiles, values, 'r-o', linewidth=2)
            ax6.set_xlabel('Percentile')
            ax6.set_ylabel('Latency (ms)')
            ax6.set_title('Latency Percentiles')
            ax6.grid(True, alpha=0.3)
        
        # 7. Summary statistics (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        if len(results) > 1:
            all_times = [r.mean_time for r in results]
            summary_text = (
                f"Runs: {len(results)}\n"
                f"Best: {min(all_times):.4f}s\n"
                f"Worst: {max(all_times):.4f}s\n"
                f"Average: {sum(all_times)/len(all_times):.4f}s\n"
                f"Improvement: {((all_times[0]-all_times[-1])/all_times[0]*100):.1f}%"
            )
            ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle(f'Performance Dashboard - {latest.name}', fontsize=16)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def save_all_figures(self, directory: Path, format: str = 'png', dpi: int = 150):
        """Save all generated figures to directory."""
        directory.mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(self.figures):
            filename = directory / f"figure_{i+1}.{format}"
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def close_all(self):
        """Close all figures to free memory."""
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()