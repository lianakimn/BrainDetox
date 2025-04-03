import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ranksums

def Metrics(control_dir=None, sweeping_dir=None, threshold=5):
    """
    Reads csv files from the Control and Sweeping groups, calculates the
    time-to-peak and peak-to-baseline metrics based on the 'Signal Difference Ratio (%)' column,
    and performs a Wilcoxon ranksum test between the groups.
    
    Parameters:
        control_dir (str): Directory for the Control group data.
                           Default: r'D:\extSSD_backup\Glymphatic_BrainSweeping_Data\Results\control data'
        sweeping_dir (str): Directory for the Sweeping group data.
                            Default: r'D:\extSSD_backup\Glymphatic_BrainSweeping_Data\Results\Sweeping data'
        threshold (float): Threshold for the signal difference ratio after the peak (default = 5).
    
    Returns:
        results (dict): Dictionary containing calculated metrics and test results:
            - control_time_to_peak_indices (list)
            - control_peak_to_baseline_indices (list)
            - sweeping_time_to_peak_indices (list)
            - sweeping_peak_to_baseline_indices (list)
            - pvalue_ttp (float): p-value for the time-to-peak indices comparison
            - pvalue_ptb (float): p-value for the peak-to-baseline indices comparison
    """
    
    # Set default directories if not provided
    if control_dir is None:
        control_dir = r'D:\extSSD_backup\Glymphatic_BrainSweeping_Data\Results\control data'
    if sweeping_dir is None:
        sweeping_dir = r'D:\extSSD_backup\Glymphatic_BrainSweeping_Data\Results\Sweeping data'
    
    # Check if directories exist
    if not os.path.exists(control_dir):
        raise FileNotFoundError(f"Control directory not found: {control_dir}")
    if not os.path.exists(sweeping_dir):
        raise FileNotFoundError(f"Sweeping directory not found: {sweeping_dir}")
    
    # Define time points (each row corresponds to a time point)
    time_points = [f"{i:02d}. {label}" for i, label in enumerate(["Pre", "0.5h"] + [f"{j}h" for j in range(1, 25)])]
    print(f"Time Points: {time_points}")
    
    def process_file(filepath, time_points, threshold):
        """
        Reads a csv file and calculates the time-to-peak and peak-to-baseline metrics 
        based on the 'Signal Difference Ratio (%)' column.
        
        Parameters:
          filepath (str): Path to the csv file.
          time_points (list): List of time point labels.
          threshold (float): Threshold value to determine the baseline (signal difference ratio falls below this value).
        
        Returns:
          time_to_peak_index (int): Index of the peak.
          time_to_peak (str): Time point corresponding to the peak.
          peak_to_baseline_index (int): Time difference (in index units) from the peak to baseline.
          peak_to_baseline (str): Time point label when the signal first falls below the threshold after the peak.
        """
        # Read csv file
        df = pd.read_csv(filepath)
        
        # Extract 'Signal Difference Ratio (%)' column values
        sdr = df['Signal Difference Ratio (%)'].values
        
        # Warn if the data length does not match the time_points length
        if len(sdr) != len(time_points):
            print(f"Warning: Length mismatch in file {filepath} -> data length: {len(sdr)}, time_points length: {len(time_points)}")
        
        # Time-to-peak: Find the index of the maximum signal difference ratio
        time_to_peak_index = np.argmax(sdr)
        time_to_peak = time_points[time_to_peak_index]
        
        # Peak-to-baseline: Find the first point after the peak where sdr falls below the threshold
        post_peak = sdr[time_to_peak_index:]
        below_threshold = np.where(post_peak < threshold)[0]
        if below_threshold.size > 0:
            relative_index = below_threshold[0]
        else:
            relative_index = len(post_peak) - 1

        peak_to_baseline_index = relative_index
        peak_to_baseline = time_points[time_to_peak_index + relative_index]
        
        return time_to_peak_index, time_to_peak, peak_to_baseline_index, peak_to_baseline
    
    # Initialize lists to store results for each group
    control_time_to_peak_indices = []
    control_peak_to_baseline_indices = []
    sweeping_time_to_peak_indices = []
    sweeping_peak_to_baseline_indices = []
    
    # Process Control group: only files starting with "Signal_Results" are used
    control_files = glob.glob(os.path.join(control_dir, "Signal_Results*.csv"))
    print(f"Number of Control group files: {len(control_files)}")
    for file in control_files:
        ttp_idx, ttp, ptb_idx, ptb = process_file(file, time_points, threshold)
        control_time_to_peak_indices.append(ttp_idx)
        control_peak_to_baseline_indices.append(ptb_idx)
        print(f"Control file: {os.path.basename(file)} | time-to-peak: {ttp} (index {ttp_idx}), peak-to-baseline: {ptb} (Δindex {ptb_idx})")
    
    # Process Sweeping group: only files starting with "Signal_Results" are used
    sweeping_files = glob.glob(os.path.join(sweeping_dir, "Signal_Results*.csv"))
    print(f"Number of Sweeping group files: {len(sweeping_files)}")
    for file in sweeping_files:
        ttp_idx, ttp, ptb_idx, ptb = process_file(file, time_points, threshold)
        sweeping_time_to_peak_indices.append(ttp_idx)
        sweeping_peak_to_baseline_indices.append(ptb_idx)
        print(f"Sweeping file: {os.path.basename(file)} | time-to-peak: {ttp} (index {ttp_idx}), peak-to-baseline: {ptb} (Δindex {ptb_idx})")
    
    # Perform Wilcoxon ranksum test
    stat_ttp, pvalue_ttp = ranksums(control_time_to_peak_indices, sweeping_time_to_peak_indices)
    stat_ptb, pvalue_ptb = ranksums(control_peak_to_baseline_indices, sweeping_peak_to_baseline_indices)
    
    print("\n=== Wilcoxon Ranksum Test Results ===")
    print(f"Time-to-peak indices comparison p-value: {pvalue_ttp}")
    print(f"Peak-to-baseline indices comparison p-value: {pvalue_ptb}")
    
    # Build results dictionary
    results = {
        'control_time_to_peak_indices': control_time_to_peak_indices,
        'control_peak_to_baseline_indices': control_peak_to_baseline_indices,
        'sweeping_time_to_peak_indices': sweeping_time_to_peak_indices,
        'sweeping_peak_to_baseline_indices': sweeping_peak_to_baseline_indices,
        'pvalue_ttp': pvalue_ttp,
        'pvalue_ptb': pvalue_ptb
    }
    
    return results

if __name__ == '__main__':
    # When running directly, call the Metrics() function and print the results.
    results = Metrics()
    print("\n=== Final Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
