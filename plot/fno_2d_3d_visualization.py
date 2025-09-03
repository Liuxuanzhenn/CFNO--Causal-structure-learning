#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO 2D/3D Visualization for Time-Frequency Domain Transformation
================================================================

This script provides advanced 2D and 3D visualizations for understanding 
how Fourier Neural Operators (FNO) transform data from time domain to 
frequency domain and back.

Features:
- 2D heatmaps of frequency spectrum evolution
- 3D surface plots of spectral analysis
- Time-frequency spectrograms
- 3D waterfall plots showing temporal evolution
- Interactive 3D visualizations

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Set matplotlib to use default fonts (English)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class Enhanced2D3DVisualizer:
    """Enhanced visualizer for 2D and 3D FNO analysis"""
    
    def __init__(self, signal_length=128, n_modes=16, n_timesteps=20):
        self.signal_length = signal_length
        self.n_modes = n_modes
        self.n_timesteps = n_timesteps
        
        # Generate time and frequency grids
        self.time_grid = np.linspace(0, 4*np.pi, signal_length)
        self.freq_grid = np.fft.fftfreq(signal_length, 1.0)[:signal_length//2+1]
        
        # Create synthetic multi-step data
        self.generate_evolution_data()
    
    def generate_evolution_data(self):
        """Generate synthetic data showing temporal evolution"""
        self.time_signals = []
        self.freq_spectra = []
        self.fno_filtered = []
        
        for t_step in range(self.n_timesteps):
            # Create evolving signal
            phase_shift = t_step * 0.3
            frequency_mod = 1 + 0.5 * np.sin(t_step * 0.2)
            
            # Multi-component signal with temporal evolution
            signal = (np.sin(self.time_grid + phase_shift) * np.exp(-t_step * 0.05) +
                     0.5 * np.sin(3 * self.time_grid * frequency_mod + phase_shift) +
                     0.3 * np.sin(8 * self.time_grid + 2 * phase_shift) +
                     0.1 * np.random.randn(self.signal_length))
            
            # Compute FFT
            fft_signal = np.fft.rfft(signal)
            spectrum = np.abs(fft_signal)
            
            # Apply FNO-style filtering (keep only low frequencies)
            filtered_fft = fft_signal.copy()
            filtered_fft[self.n_modes:] *= 0.1  # Strongly attenuate high frequencies
            filtered_spectrum = np.abs(filtered_fft)
            
            self.time_signals.append(signal)
            self.freq_spectra.append(spectrum)
            self.fno_filtered.append(filtered_spectrum)
        
        # Convert to numpy arrays
        self.time_signals = np.array(self.time_signals)
        self.freq_spectra = np.array(self.freq_spectra)
        self.fno_filtered = np.array(self.fno_filtered)

    def create_2d_heatmap_visualization(self):
        """Create 2D heatmap visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FNO 2D Visualization: Time-Frequency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time domain evolution heatmap
        im1 = axes[0, 0].imshow(self.time_signals, aspect='auto', cmap='viridis', 
                               extent=[0, 4*np.pi, self.n_timesteps, 0])
        axes[0, 0].set_title('Time Domain Signal Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Time Step')
        plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')
        
        # 2. Full frequency spectrum evolution
        im2 = axes[0, 1].imshow(self.freq_spectra, aspect='auto', cmap='plasma',
                               extent=[0, len(self.freq_grid), self.n_timesteps, 0])
        axes[0, 1].set_title('Full Frequency Spectrum Evolution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Frequency Mode Index')
        axes[0, 1].set_ylabel('Time Step')
        plt.colorbar(im2, ax=axes[0, 1], label='Spectral Magnitude')
        
        # 3. FNO filtered spectrum evolution
        im3 = axes[1, 0].imshow(self.fno_filtered, aspect='auto', cmap='magma',
                               extent=[0, len(self.freq_grid), self.n_timesteps, 0])
        axes[1, 0].set_title('FNO Filtered Spectrum Evolution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Frequency Mode Index')
        axes[1, 0].set_ylabel('Time Step')
        plt.colorbar(im3, ax=axes[1, 0], label='Filtered Magnitude')
        
        # 4. Filtering effect comparison (difference)
        filtering_effect = self.freq_spectra - self.fno_filtered
        im4 = axes[1, 1].imshow(filtering_effect, aspect='auto', cmap='RdBu_r',
                               extent=[0, len(self.freq_grid), self.n_timesteps, 0])
        axes[1, 1].set_title('FNO Filtering Effect (Removed Components)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Frequency Mode Index')
        axes[1, 1].set_ylabel('Time Step')
        plt.colorbar(im4, ax=axes[1, 1], label='Removed Magnitude')
        
        plt.tight_layout()
        return fig

    def create_3d_surface_visualization(self):
        """Create 3D surface visualizations"""
        fig = plt.figure(figsize=(20, 15))
        
        # Create meshgrids for 3D plotting
        T, F = np.meshgrid(range(self.n_timesteps), range(len(self.freq_grid)))
        
        # 1. 3D surface of full spectrum evolution
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(T, F, self.freq_spectra.T, cmap='viridis', alpha=0.8)
        ax1.set_title('3D Full Frequency Spectrum Evolution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Frequency Mode')
        ax1.set_zlabel('Spectral Magnitude')
        ax1.view_init(elev=30, azim=45)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        # 2. 3D surface of FNO filtered spectrum
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(T, F, self.fno_filtered.T, cmap='plasma', alpha=0.8)
        ax2.set_title('3D FNO Filtered Spectrum Evolution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Frequency Mode')
        ax2.set_zlabel('Filtered Magnitude')
        ax2.view_init(elev=30, azim=45)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        # 3. 3D waterfall plot of frequency evolution
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        for i in range(0, self.n_timesteps, 3):  # Plot every 3rd timestep for clarity
            ax3.plot(self.freq_grid[:50], [i]*50, self.freq_spectra[i][:50], 
                    alpha=0.7, linewidth=2, label=f'Step {i}')
        ax3.set_title('3D Waterfall: Frequency Spectrum Evolution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Time Step')
        ax3.set_zlabel('Magnitude')
        ax3.view_init(elev=20, azim=60)
        
        # 4. 3D comparison: Before vs After FNO filtering
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        # Plot original spectrum (blue)
        mid_step = self.n_timesteps // 2
        ax4.plot(self.freq_grid[:30], [0]*30, self.freq_spectra[mid_step][:30], 
                'b-', linewidth=3, alpha=0.8, label='Original Spectrum')
        # Plot filtered spectrum (red)
        ax4.plot(self.freq_grid[:30], [1]*30, self.fno_filtered[mid_step][:30], 
                'r-', linewidth=3, alpha=0.8, label='FNO Filtered')
        # Add vertical lines to show filtering cutoff
        ax4.plot([self.freq_grid[self.n_modes]]*2, [0, 1], [0, max(self.freq_spectra[mid_step][:30])],
                'k--', linewidth=2, alpha=0.7, label='FNO Cutoff')
        ax4.set_title('3D Comparison: Original vs FNO Filtered', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Frequency')
        ax4.set_ylabel('Filter Type')
        ax4.set_zlabel('Magnitude')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Original', 'Filtered'])
        ax4.legend()
        
        plt.tight_layout()
        return fig

    def create_time_frequency_spectrogram(self):
        """Create time-frequency spectrogram analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FNO Time-Frequency Spectrogram Analysis', fontsize=16, fontweight='bold')
        
        # Select a representative signal for detailed analysis
        mid_signal = self.time_signals[self.n_timesteps // 2]
        
        # 1. Original signal in time domain
        axes[0, 0].plot(self.time_grid, mid_signal, 'b-', linewidth=2)
        axes[0, 0].set_title('Representative Time Domain Signal', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Frequency domain representation
        fft_signal = np.fft.rfft(mid_signal)
        axes[0, 1].plot(self.freq_grid, np.abs(fft_signal), 'r-', linewidth=2, label='Original FFT')
        axes[0, 1].plot(self.freq_grid, self.fno_filtered[self.n_timesteps // 2], 
                       'g-', linewidth=2, label='FNO Filtered')
        axes[0, 1].axvline(x=self.freq_grid[self.n_modes], color='k', linestyle='--', 
                          alpha=0.7, label='FNO Cutoff')
        axes[0, 1].set_title('Frequency Domain Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Short-Time Fourier Transform (STFT) - Spectrogram
        from scipy import signal as scipy_signal
        f_stft, t_stft, Zxx = scipy_signal.stft(mid_signal, nperseg=32, noverlap=24)
        im3 = axes[1, 0].pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud', cmap='jet')
        axes[1, 0].set_title('Short-Time Fourier Transform (STFT)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Frequency')
        plt.colorbar(im3, ax=axes[1, 0], label='Magnitude')
        
        # 4. FNO mode evolution over time
        mode_evolution = self.freq_spectra[:, :self.n_modes]
        im4 = axes[1, 1].imshow(mode_evolution.T, aspect='auto', cmap='viridis',
                               extent=[0, self.n_timesteps, self.n_modes, 0])
        axes[1, 1].set_title(f'FNO Low-Frequency Modes Evolution (First {self.n_modes} modes)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Frequency Mode Index')
        plt.colorbar(im4, ax=axes[1, 1], label='Magnitude')
        
        plt.tight_layout()
        return fig

    def create_interactive_3d_analysis(self):
        """Create an interactive 3D analysis with multiple perspectives"""
        fig = plt.figure(figsize=(20, 16))
        
        # Select a representative signal for detailed analysis
        mid_signal = self.time_signals[self.n_timesteps // 2]
        fft_signal = np.fft.rfft(mid_signal)
        
        # 1. 3D Volume visualization of time-frequency evolution
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # Create a 3D scatter plot showing evolution
        time_steps = np.arange(self.n_timesteps)
        freq_modes = np.arange(len(self.freq_grid))
        
        # Sample every few points for clarity
        step_sample = slice(0, self.n_timesteps, 2)
        freq_sample = slice(0, min(50, len(self.freq_grid)), 2)
        
        T_sample, F_sample = np.meshgrid(time_steps[step_sample], freq_modes[freq_sample])
        magnitudes = self.freq_spectra[step_sample][:, freq_sample].T
        
        # Create 3D scatter with color mapping
        scatter = ax1.scatter(T_sample.flatten(), F_sample.flatten(), magnitudes.flatten(),
                             c=magnitudes.flatten(), cmap='viridis', s=20, alpha=0.6)
        ax1.set_title('3D Volume: Frequency Evolution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Frequency Mode')
        ax1.set_zlabel('Magnitude')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
        
        # 2. 3D Isosurface-like visualization
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        
        # Create contour-like 3D visualization
        levels = np.linspace(0, np.max(self.freq_spectra), 5)
        for i, level in enumerate(levels[1:]):  # Skip zero level
            # Find points above this threshold
            mask = self.freq_spectra > level
            if np.any(mask):
                y_coords, x_coords = np.where(mask)
                z_coords = self.freq_spectra[mask]
                ax2.scatter(x_coords, y_coords, z_coords, 
                           alpha=0.3 + i*0.15, s=10, label=f'Level {level:.2f}')
        
        ax2.set_title('3D Threshold Analysis', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frequency Mode')
        ax2.set_ylabel('Time Step')
        ax2.set_zlabel('Magnitude')
        ax2.legend()
        
        # 3. 3D Network-like visualization showing FNO operation
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        
        # Show time domain, frequency domain, and filtered domain as layers
        z_time = 0
        z_freq = 1
        z_filtered = 2
        
        # Time domain layer
        sample_indices = np.linspace(0, len(self.time_grid)-1, 30, dtype=int)
        ax3.plot(self.time_grid[sample_indices], 
                [z_time]*len(sample_indices), 
                mid_signal[sample_indices], 
                'b-', linewidth=3, alpha=0.8, label='Time Domain')
        
        # Frequency domain layer
        freq_sample_indices = np.linspace(0, len(self.freq_grid)-1, 30, dtype=int)
        ax3.plot(self.freq_grid[freq_sample_indices], 
                [z_freq]*len(freq_sample_indices), 
                np.abs(fft_signal)[freq_sample_indices], 
                'r-', linewidth=3, alpha=0.8, label='Frequency Domain')
        
        # Filtered domain layer
        ax3.plot(self.freq_grid[freq_sample_indices], 
                [z_filtered]*len(freq_sample_indices), 
                self.fno_filtered[self.n_timesteps // 2][freq_sample_indices], 
                'g-', linewidth=3, alpha=0.8, label='FNO Filtered')
        
        # Add transformation arrows
        for i in range(0, len(sample_indices), 5):
            ax3.plot([self.time_grid[sample_indices[i]], self.freq_grid[freq_sample_indices[i]]], 
                    [z_time, z_freq],
                    [mid_signal[sample_indices[i]], np.abs(fft_signal)[freq_sample_indices[i]]],
                    'k--', alpha=0.3)
        
        ax3.set_title('3D FNO Transformation Pipeline', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time/Frequency')
        ax3.set_ylabel('Processing Stage')
        ax3.set_zlabel('Amplitude')
        ax3.set_yticks([z_time, z_freq, z_filtered])
        ax3.set_yticklabels(['Time', 'Frequency', 'Filtered'])
        ax3.legend()
        
        # 4-6. Multiple viewing angles of the same data
        viewing_angles = [(30, 45), (60, 90), (15, 135)]
        titles = ['3D View: Elevation 30°', '3D View: Elevation 60°', '3D View: Elevation 15°']
        
        # Create meshgrids for 3D plotting
        T, F = np.meshgrid(range(self.n_timesteps), range(len(self.freq_grid)))
        
        for i, ((elev, azim), title) in enumerate(zip(viewing_angles, titles)):
            ax = fig.add_subplot(2, 3, 4+i, projection='3d')
            surf = ax.plot_surface(T, F, self.freq_spectra.T, cmap='coolwarm', alpha=0.7)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Frequency Mode')
            ax.set_zlabel('Magnitude')
            ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        return fig

    def save_analysis_data(self):
        """Save detailed analysis data to CSV files"""
        print("Saving analysis data...")
        
        # Save time domain signals
        time_df = pd.DataFrame(self.time_signals.T)
        time_df.columns = [f'TimeStep_{i}' for i in range(self.n_timesteps)]
        time_df.index = self.time_grid
        time_df.to_csv('fno_time_evolution.csv')
        
        # Save frequency domain analysis
        freq_df = pd.DataFrame({
            'Frequency': self.freq_grid,
            'FullSpectrum_Avg': np.mean(self.freq_spectra, axis=0),
            'FNOFiltered_Avg': np.mean(self.fno_filtered, axis=0),
            'FilteringEffect_Avg': np.mean(self.freq_spectra - self.fno_filtered, axis=0)
        })
        freq_df.to_csv('fno_frequency_analysis.csv', index=False)
        
        # Save temporal evolution of key frequency modes
        mode_evolution_df = pd.DataFrame(self.freq_spectra[:, :self.n_modes])
        mode_evolution_df.columns = [f'Mode_{i}' for i in range(self.n_modes)]
        mode_evolution_df.index = [f'TimeStep_{i}' for i in range(self.n_timesteps)]
        mode_evolution_df.to_csv('fno_mode_evolution.csv')
        
        print("✓ Data saved:")
        print("  • fno_time_evolution.csv - Time domain evolution")
        print("  • fno_frequency_analysis.csv - Frequency domain analysis")
        print("  • fno_mode_evolution.csv - Key mode temporal evolution")

def main():
    """Main function to generate all visualizations"""
    print("="*80)
    print("FNO 2D/3D Visualization Generator")
    print("="*80)
    
    # Create visualizer instance
    print("1. Initializing visualizer...")
    visualizer = Enhanced2D3DVisualizer(signal_length=128, n_modes=16, n_timesteps=20)
    
    # Generate 2D heatmap visualization
    print("2. Creating 2D heatmap visualizations...")
    fig_2d = visualizer.create_2d_heatmap_visualization()
    fig_2d.savefig('fno_2d_heatmap_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    print("✓ 2D heatmap visualization saved: fno_2d_heatmap_analysis.png")
    
    # Generate 3D surface visualization
    print("3. Creating 3D surface visualizations...")
    fig_3d_surface = visualizer.create_3d_surface_visualization()
    fig_3d_surface.savefig('fno_3d_surface_analysis.png', dpi=300, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
    print("✓ 3D surface visualization saved: fno_3d_surface_analysis.png")
    
    # Generate time-frequency spectrogram
    print("4. Creating time-frequency spectrogram...")
    fig_spectrogram = visualizer.create_time_frequency_spectrogram()
    fig_spectrogram.savefig('fno_spectrogram_analysis.png', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
    print("✓ Spectrogram analysis saved: fno_spectrogram_analysis.png")
    
    # Generate interactive 3D analysis
    print("5. Creating interactive 3D analysis...")
    fig_interactive = visualizer.create_interactive_3d_analysis()
    fig_interactive.savefig('fno_interactive_3d_analysis.png', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
    print("✓ Interactive 3D analysis saved: fno_interactive_3d_analysis.png")
    
    # Save analysis data
    print("6. Saving analysis data...")
    visualizer.save_analysis_data()
    
    # Display all plots
    print("7. Displaying visualizations...")
    plt.show()
    
    print("\n" + "="*80)
    print("✅ FNO 2D/3D Visualization Complete!")
    print("\nGenerated Files:")
    print("  📊 fno_2d_heatmap_analysis.png - 2D heatmap evolution")
    print("  🏔️  fno_3d_surface_analysis.png - 3D surface plots")
    print("  📈 fno_spectrogram_analysis.png - Time-frequency analysis")
    print("  🌀 fno_interactive_3d_analysis.png - Interactive 3D views")
    print("  📋 fno_*.csv - Detailed analysis data")
    print("\nKey Insights:")
    print("  • 2D heatmaps show temporal evolution of frequency content")
    print("  • 3D surfaces reveal the full time-frequency landscape")
    print("  • Spectrograms provide detailed time-frequency resolution")
    print("  • Interactive views show FNO transformation pipeline")
    print("="*80)

if __name__ == "__main__":
    try:
        # Check for required packages
        required_packages = ['numpy', 'matplotlib', 'pandas', 'torch', 'scipy']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing required packages: {', '.join(missing_packages)}")
            print("Please install them using: pip install " + " ".join(missing_packages))
        else:
            main()
            
    except KeyboardInterrupt:
        print("\n❌ Visualization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc() 