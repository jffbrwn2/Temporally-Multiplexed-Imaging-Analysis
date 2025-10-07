#!/usr/bin/env python3
"""
Python implementation of the MATLAB unmixing script for fluorescent protein demultiplexing.
Reimplemented from Unmixing_25_07_24.m

This script performs linear unmixing of fluorescence curves to separate contributions
from different fluorescent proteins with unique decay kinetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize
import os
import glob
import re
from pathlib import Path
import time
import tifffile
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class FluorescenceUnmixer:
    """
    A class to perform fluorescence unmixing using reference decay curves.
    """
    
    def __init__(self, image_width=1360, image_height=1360, num_colors=5, use_fitted_curves=False, constrain_background=False):
        """
        Initialize the unmixer with image dimensions and number of colors.
        
        Parameters:
        -----------
        image_width : int
            Width of the images
        image_height : int  
            Height of the images
        num_colors : int
            Number of fluorescent colors/channels
        use_fitted_curves : bool
            If True, use datRefK.csv (fitted exponential curves)
            If False, use datRef.csv (population averaged curves)
        constrain_background : bool
            If True, constrain EGFP channel (background) to 0-500 range
            If False, use standard NNLS (non-negative only)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.num_colors = num_colors
        self.use_fitted_curves = use_fitted_curves
        self.constrain_background = constrain_background
        self.datRef = None
        self.sc_curves = None
        
    def load_reference_data(self, ref_path_base):
        """
        Load and normalize reference curve data.
        
        Parameters:
        -----------
        ref_path_base : str
            Base path to the reference CSV files (without extension)
            Will automatically select datRef.csv or datRefK.csv based on use_fitted_curves
        """
        # Determine which reference file to use
        if self.use_fitted_curves:
            ref_path = ref_path_base.replace('datRef.csv', 'datRefK.csv')
            curve_type = "fitted exponential curves (datRefK.csv)"
        else:
            ref_path = ref_path_base if ref_path_base.endswith('.csv') else ref_path_base + '.csv'
            if 'datRefK' in ref_path:
                ref_path = ref_path.replace('datRefK.csv', 'datRef.csv')
            curve_type = "population averaged curves (datRef.csv)"
            
        print(f"Loading reference data from: {ref_path}")
        print(f"Using: {curve_type}")
        
        # Check if file exists
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference file not found: {ref_path}")
        
        # Load reference data
        self.datRef = pd.read_csv(ref_path, sep='\t').values
        
        # Normalize reference curves (equivalent to MATLAB: datRef ./ max(datRef, [], 1))
        self.sc_curves = self.datRef / np.max(self.datRef, axis=0, keepdims=True)
        
        print(f"Reference data shape: {self.datRef.shape}")
        print(f"Normalized curves shape: {self.sc_curves.shape}")
        
        # Print some statistics about the curves
        print(f"Reference curve statistics:")
        channel_labels = ['AP1-EGFP', 'DUSP5-Skylan', 'EGR1-Dronpa', 'FOS-GreenFast', 'SARE-rsFastLime']
        for i, label in enumerate(channel_labels):
            final_val = self.sc_curves[-1, i]  # Final normalized value
            print(f"  {label}: final value = {final_val:.3f}")
        
    def find_matching_files(self, green_folder, red_mask_folder):
        """
        Find matching green curve and red mask files.
        
        Parameters:
        -----------
        green_folder : str
            Path to folder containing green curve .mat files
        red_mask_folder : str
            Path to folder containing red mask .mat files
            
        Returns:
        --------
        list of tuples
            (well_id, roi, green_file_path, red_file_path)
        """
        print("Finding matching files...")
        
        # Get all green curve files
        green_pattern = os.path.join(green_folder, "*green_curve_2.mat")
        green_files = glob.glob(green_pattern)
        
        # Get all red mask files  
        red_pattern = os.path.join(red_mask_folder, "*red_mask.mat")
        red_files = glob.glob(red_pattern)
        
        print(f"Found {len(green_files)} green curve files")
        print(f"Found {len(red_files)} red mask files")
        
        # Extract well and ROI information
        file_matches = []
        green_regex = r'([A-Z]\d+)_ROI_(\d+)_green_curve_2\.mat'
        
        for green_file in green_files:
            green_basename = os.path.basename(green_file)
            match = re.search(green_regex, green_basename)
            
            if match:
                well_id = match.group(1)
                roi = int(match.group(2))
                
                # Find corresponding red mask file
                red_pattern = f"{well_id}_ROI_{roi}_red_mask.mat"
                red_file = os.path.join(red_mask_folder, red_pattern)
                
                if os.path.exists(red_file):
                    file_matches.append((well_id, roi, green_file, red_file))
                else:
                    print(f"Warning: No matching red mask for {green_basename}")
        
        print(f"Found {len(file_matches)} matching file pairs")
        return file_matches
    
    def objective_function(self, coeffs, sc_curves, target_curve):
        """
        Objective function for optimization - minimizes L2 norm between 
        predicted and target curves.
        
        Parameters:
        -----------
        coeffs : numpy.ndarray
            Coefficients for linear combination
        sc_curves : numpy.ndarray
            Normalized reference curves
        target_curve : numpy.ndarray
            Target curve to fit
            
        Returns:
        --------
        float
            L2 norm of the residual
        """
        predicted = np.dot(sc_curves, coeffs)
        return np.linalg.norm(predicted - target_curve)
    
    def load_mat_with_retry(self, filepath, max_retries=3, delay=1.0, timeout=5.0):
        """
        Load MAT file with retry logic and timeout for NFS/network issues.
        
        Parameters:
        -----------
        filepath : str
            Path to the .mat file
        max_retries : int
            Maximum number of retry attempts
        delay : float
            Delay between retries in seconds
        timeout : float
            Timeout for each load attempt in seconds
            
        Returns:
        --------
        dict
            Loaded MAT file data
        """
        import time
        import signal
        from contextlib import contextmanager
        
        @contextmanager
        def timeout_handler(seconds):
            def timeout_error(signum, frame):
                raise TimeoutError(f"File load timed out after {seconds} seconds")
            
            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_error)
            signal.alarm(int(seconds))
            
            try:
                yield
            finally:
                # Restore the old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        for attempt in range(max_retries):
            try:
                print(f"  Loading {os.path.basename(filepath)} (attempt {attempt + 1}/{max_retries})...")
                
                with timeout_handler(timeout):
                    data = sio.loadmat(filepath)
                    print(f"  Successfully loaded {os.path.basename(filepath)}")
                    return data
                    
            except (OSError, IOError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    print(f"  Warning: File load failed (attempt {attempt + 1}), retrying in {delay}s...")
                    print(f"  Error: {str(e)}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"  Failed to load {filepath} after {max_retries} attempts")
                    raise e
    
    def unmix_single_pixel(self, target_curve, initial_guess=None):
        """
        Unmix a single pixel's time series curve using either NNLS or bounded optimization.
        
        Parameters:
        -----------
        target_curve : numpy.ndarray
            Time series curve for a single pixel (raw intensities)
        initial_guess : numpy.ndarray, optional
            Initial guess for coefficients
            
        Returns:
        --------
        numpy.ndarray
            Optimized coefficients
        """
        # Use raw target curve (no normalization)
        # Use raw reference curves (no normalization)
        raw_reference_curves = self.datRef  # Use original raw reference data
        
        if self.constrain_background:
            # Bounded optimization: EGFP channel (index 0) constrained to 0-500 for background estimation
            from scipy.optimize import minimize
            
            # Define objective function for least squares
            def objective(coeffs):
                predicted = np.dot(raw_reference_curves, coeffs)
                return np.sum((predicted - target_curve) ** 2)
            
            # Set up bounds: EGFP (channel 0) constrained to 0-500, others non-negative
            bounds = [(0, 500)]  # EGFP background constraint
            for i in range(1, raw_reference_curves.shape[1]):
                bounds.append((0, None))  # Other channels non-negative only
            
            # Initial guess
            if initial_guess is None:
                initial_guess = np.ones(raw_reference_curves.shape[1]) * 0.1
                initial_guess[0] = 250  # Start EGFP at middle of range
            
            # Optimize with bounds
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if not result.success:
                print(f"Warning: Optimization failed: {result.message}")
            
            return result.x
        else:
            # Standard NNLS approach (original method)
            from scipy.optimize import nnls
            
            # Non-negative least squares: minimize ||Ax - b||^2 subject to x >= 0
            # where A is the reference curves matrix, x is coefficients, b is target curve
            coefficients, residual = nnls(raw_reference_curves, target_curve)
            
            return coefficients
    
    def process_single_sample(self, well_id, roi, green_file, red_file, output_dir):
        """
        Process a single well/ROI combination.
        
        Parameters:
        -----------
        well_id : str
            Well identifier
        roi : int
            ROI number
        green_file : str
            Path to green curve file
        red_file : str
            Path to red mask file
        output_dir : str
            Output directory
        """
        print(f"Processing {well_id} - ROI {roi}")
        
        try:
            # Load data files with retry logic for NFS issues
            green_data = self.load_mat_with_retry(green_file)
            red_data = self.load_mat_with_retry(red_file)
            
            # Extract data (assuming 'datPic' and 'red_image' variable names)
            if 'datPic' in green_data:
                stack = green_data['datPic'].astype(np.float64)
            else:
                # Try other common variable names
                possible_names = [k for k in green_data.keys() if not k.startswith('__')]
                stack = green_data[possible_names[0]].astype(np.float64)
            
            if 'red_image' in red_data:
                active_area = red_data['red_image']
            else:
                possible_names = [k for k in red_data.keys() if not k.startswith('__')]
                active_area = red_data[possible_names[0]]
            
            print(f"Stack shape: {stack.shape}")
            print(f"Active area shape: {active_area.shape}")
            
            # Find active pixels
            active_coords = np.where(active_area)
            num_active_pixels = len(active_coords[0])
            print(f"Number of active pixels: {num_active_pixels}")
            
            if num_active_pixels == 0:
                print(f"No active pixels found for {well_id}-ROI{roi}")
                return
            
            # Process at cell/ROI level - average across all active pixels first
            print(f"Computing cell-level average curve from {num_active_pixels} pixels...")
            
            # Extract time series for all active pixels and average them
            cell_time_series = np.zeros(stack.shape[0])  # 70 timeframes
            
            for i in range(num_active_pixels):
                y, x = active_coords[0][i], active_coords[1][i]
                cell_time_series += stack[:, y, x]
            
            # Average across all active pixels to get cell-level curve
            cell_time_series = cell_time_series / num_active_pixels
            
            print(f"Cell-level curve: min={np.min(cell_time_series):.1f}, max={np.max(cell_time_series):.1f}")
            
            # Unmix the cell-level curve
            print("Unmixing cell-level curve...")
            cell_coeffs = self.unmix_single_pixel(cell_time_series)
            
            print(f"Cell-level coefficients: {[f'{c:.3f}' for c in cell_coeffs]}")
            
            # Store results - assign same coefficients to all active pixels for visualization
            comp_img = np.zeros((self.num_colors, self.image_height, self.image_width))
            for i in range(num_active_pixels):
                y, x = active_coords[0][i], active_coords[1][i]
                comp_img[:, y, x] = cell_coeffs
            
            # Export results
            self.export_results(well_id, roi, comp_img, stack, output_dir)
            
        except Exception as e:
            print(f"Error processing {well_id}-ROI{roi}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_results(self, well_id, roi, comp_img, stack, output_dir):
        """
        Export unmixing results in various formats.
        
        Parameters:
        -----------
        well_id : str
            Well identifier
        roi : int
            ROI number
        comp_img : numpy.ndarray
            Compensation image (num_colors x height x width)
        stack : numpy.ndarray
            Original image stack
        output_dir : str
            Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Channel labels mapping promoter-fluorophore pairs
        channel_labels = [
            'Constant (AP1-EGFP XOR Baseline)',
            'DUSP5-Skylan', 
            'EGR1-Dronpa',
            'FOS-GreenFast',
            'SARE-rsFastLime'
        ]
        
        # Add reference type and background constraint to title
        ref_type = "Fitted" if self.use_fitted_curves else "Population Avg"
        bg_constraint = " + BG Constrain" if self.constrain_background else ""
        method_label = ref_type + bg_constraint
        
        # Create comprehensive figure with bar plot and sample analysis
        fig = plt.figure(figsize=(16, 8))
        
        # Bar plot (left panel) - now showing cell-level coefficients
        ax1 = plt.subplot(1, 2, 1)
        # Since all pixels have the same coefficients now, just get one pixel's values
        active_coords = np.where(np.sum(comp_img, axis=0) > 0)
        if len(active_coords[0]) > 0:
            y_sample, x_sample = active_coords[0][0], active_coords[1][0]
            bar_data = comp_img[:, y_sample, x_sample]  # Cell-level coefficients
        else:
            bar_data = np.zeros(self.num_colors)
            
        bars = ax1.bar(range(self.num_colors), bar_data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_xlabel('Channel', fontsize=12)
        ax1.set_ylabel('Unmixed Coefficient (Non-Negative)', fontsize=12)
        ax1.set_title(f'Cell-Level Unmixing Results ({method_label})', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(self.num_colors))
        ax1.set_xticklabels(channel_labels, rotation=45, ha='right')
        
        # Dynamic y-axis limit based on data
        max_coeff = np.max(bar_data) if np.max(bar_data) > 0 else 1.0
        ax1.set_ylim(0, max_coeff * 1.1)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, bar_data):
            if value > 0:  # Only label non-zero values
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_coeff * 0.02, 
                        f'{value:.2e}' if value < 0.01 else f'{value:.3f}', 
                        ha='center', va='bottom', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add total coefficient sum (no longer constrained to 1)
        coeff_sum = np.sum(bar_data)
        ax1.text(0.02, 0.98, f'Total: {coeff_sum:.2e}', transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Sample curve analysis (right panel)
        ax2 = plt.subplot(1, 2, 2)
        
        # Show the cell-level analysis
        active_coords = np.where(np.sum(comp_img, axis=0) > 0)
        frames = np.arange(self.datRef.shape[0])
        
        if len(active_coords[0]) > 0:
            # Compute cell-level average curve (same as used for unmixing)
            cell_time_series = np.zeros(stack.shape[0])
            num_active_pixels = len(active_coords[0])
            
            for i in range(num_active_pixels):
                y, x = active_coords[0][i], active_coords[1][i]
                cell_time_series += stack[:, y, x]
            
            # Average across all active pixels to get cell-level curve
            cell_curve_raw = cell_time_series / num_active_pixels
            
            # Get cell-level coefficients (same for all pixels)
            y_sample, x_sample = active_coords[0][0], active_coords[1][0]
            cell_coeffs = comp_img[:, y_sample, x_sample]
            
            # Compute fitted curve using raw reference curves and coefficients
            fitted_curve_raw = np.dot(self.datRef, cell_coeffs)
            
            # Plot both curves (RAW intensities) - now cell-level
            ax2.plot(frames, cell_curve_raw, 'ko-', label='Measured (Cell Avg)', alpha=0.7, markersize=4)
            ax2.plot(frames, fitted_curve_raw, 'r--', label='Fitted (NNLS)', alpha=0.8, linewidth=2)
            
            # Show individual component contributions (raw intensity)
            for i, (label, coeff, color) in enumerate(zip(channel_labels, cell_coeffs, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])):
                if coeff > np.max(cell_coeffs) * 0.01:  # Show contributions > 1% of max
                    component_curve_raw = self.datRef[:, i] * coeff
                    ax2.plot(frames, component_curve_raw, ':', color=color, alpha=0.6, linewidth=1.5, 
                            label=f'{label} ({coeff:.2e})')
            
            ax2.set_xlabel('Frame', fontsize=12)
            ax2.set_ylabel('Raw Fluorescence Intensity (a.u.)', fontsize=12)
            ax2.set_title(f'Cell-Level Analysis (Raw Data)\n{num_active_pixels} pixels averaged', fontsize=14, fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Calculate and display fit quality using raw intensities
            residual_raw = np.linalg.norm(fitted_curve_raw - cell_curve_raw)
            residual_norm = residual_raw / np.max(cell_curve_raw)  # Normalized residual
            
            # Add SNR information
            signal_max = np.max(cell_curve_raw)
            noise_std = np.std(cell_curve_raw[-10:])  # Estimate noise from last 10 frames
            snr = signal_max / noise_std if noise_std > 0 else np.inf
            
            info_text = f'NNLS Residual: {residual_raw:.0f}\nNorm. Residual: {residual_norm:.3f}\nSNR: {snr:.1f}\nMax Intensity: {signal_max:.0f}\nPixels: {num_active_pixels}'
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'No active pixels found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Sample Pixel Analysis', fontsize=14, fontweight='bold')
        
        # Main title
        fig.suptitle(f'Fluorophore Unmixing Analysis: {well_id} - ROI {roi}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the comprehensive plot
        comprehensive_filename = f"{well_id}-ROI{roi}-comprehensive.png"
        plt.savefig(os.path.join(output_dir, comprehensive_filename), dpi=200, bbox_inches='tight')
        plt.close()
        
        # Also save just the bar plot for compatibility
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(self.num_colors), bar_data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Total Signal (Unmixed Intensity)', fontsize=12)
        plt.title(f'Fluorophore Unmixing Results ({method_label}): {well_id} - ROI {roi}', fontsize=14, fontweight='bold')
        plt.xticks(range(self.num_colors), channel_labels, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar, value in zip(bars, bar_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bar_data)*0.01, 
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        bar_filename = f"{well_id}-ROI{roi}-barplot.png"
        plt.savefig(os.path.join(output_dir, bar_filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create compensated channels
        chan = np.zeros((self.num_colors, self.image_height, self.image_width))
        for c in range(self.num_colors):
            chan[c] = comp_img[c] * stack[0]  # Use first frame as reference
        
        # Multi-channel TIFF
        multi_chan_stack = np.zeros((self.num_colors, self.image_height, self.image_width), dtype=np.uint16)
        for c in range(self.num_colors):
            # Normalize and convert to uint16
            chan_norm = chan[c] / np.max(chan[c]) if np.max(chan[c]) > 0 else chan[c]
            multi_chan_stack[c] = (chan_norm * 65535).astype(np.uint16)
        
        tiff_filename = f"{well_id}-ROI{roi}-curve2-multichannel.tif"
        tifffile.imwrite(os.path.join(output_dir, tiff_filename), multi_chan_stack)
        
        # Save compensation data as .npz (Python equivalent of .mat)
        npz_filename = f"{well_id}-ROI{roi}-c2-outMAT.npz"
        np.savez_compressed(os.path.join(output_dir, npz_filename), compImg=comp_img)
        
        print(f"Exported results for {well_id}-ROI{roi}")
    
    def run_unmixing(self, ref_path, green_folder, red_mask_folder, output_dir):
        """
        Run the complete unmixing pipeline.
        
        Parameters:
        -----------
        ref_path : str
            Path to reference curve data
        green_folder : str
            Path to green curve folder
        red_mask_folder : str
            Path to red mask folder
        output_dir : str
            Output directory
        """
        start_time = time.time()
        
        # Add suffix to output dir to distinguish between reference types and background constraint
        if self.use_fitted_curves:
            output_dir = output_dir + "_fitted_curves"
        else:
            output_dir = output_dir + "_population_avg"
            
        if self.constrain_background:
            output_dir = output_dir + "_bg_constrained"
        
        print(f"Output will be saved to: {output_dir}")
        
        # Load reference data
        self.load_reference_data(ref_path)
        
        # Find matching files
        file_matches = self.find_matching_files(green_folder, red_mask_folder)
        
        if not file_matches:
            print("No matching files found!")
            return
        
        # Process each file pair
        for well_id, roi, green_file, red_file in file_matches:
            self.process_single_sample(well_id, roi, green_file, red_file, output_dir)
        
        elapsed_time = time.time() - start_time
        print(f"\nUnmixing completed in {elapsed_time:.2f} seconds")
        print(f"Processed {len(file_matches)} samples")


def main(use_fitted_curves=False, constrain_background=False):
    """
    Main function to run the unmixing pipeline.
    
    Parameters:
    -----------
    use_fitted_curves : bool
        If True, use datRefK.csv (fitted exponential curves)
        If False, use datRef.csv (population averaged curves)
    constrain_background : bool
        If True, constrain EGFP channel (background) to 0-500 range
        If False, use standard NNLS (non-negative only)
    """
    # Updated paths for this computer
    base_path = "/Users/jbrown/Documents/boyden_lab/Python_individual_FPs_curve_fitting"
    
    ref_path = os.path.join(base_path, "datRef.csv")
    green_folder = os.path.join(base_path, "maelle-neuron-expriment-20250804", "green_mat_curves")
    red_mask_folder = os.path.join(base_path, "maelle-neuron-expriment-20250804", "red_masks_mix")
    output_dir = os.path.join(base_path, "unmixing_output")
    
    # Check if paths exist
    print("Checking file paths...")
    ref_check_path = ref_path.replace('datRef.csv', 'datRefK.csv') if use_fitted_curves else ref_path
    for path, name in [(ref_check_path, "Reference data"), (green_folder, "Green curves"), (red_mask_folder, "Red masks")]:
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (NOT FOUND)")
    
    # Initialize unmixer with chosen reference type and background constraint option
    unmixer = FluorescenceUnmixer(use_fitted_curves=use_fitted_curves, constrain_background=constrain_background)
    
    # Run unmixing
    unmixer.run_unmixing(ref_path, green_folder, red_mask_folder, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fluorescence unmixing with optional background constraint')
    parser.add_argument('--fitted', action='store_true', 
                       help='Use fitted exponential curves (datRefK.csv) instead of population averaged (datRef.csv)')
    parser.add_argument('--constrain-background', action='store_true',
                       help='Constrain EGFP channel (background) to 0-500 range')
    
    args = parser.parse_args()
    
    print(f"Running unmixing with:")
    print(f"  Reference curves: {'Fitted exponential' if args.fitted else 'Population averaged'}")
    print(f"  Background constraint: {'Enabled (0-500)' if args.constrain_background else 'Disabled (NNLS)'}")
    print()
    
    main(use_fitted_curves=args.fitted, constrain_background=args.constrain_background)