import matplotlib
from Metrics import Metrics  # Import the Metrics() function

import sys
import matplotlib

if sys.platform == 'win32':
    try:
        import PyQt5  # Try to import PyQt5 for Qt5Agg
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('TkAgg')
elif sys.platform == 'darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('Qt5Agg')

import numpy as np
import os
import csv
from scipy.stats import ranksums
import warnings
warnings.filterwarnings("ignore")    

# Directories for results
control_dir = 'D:\extSSD_backup\Glymphatic_BrainSweeping_Data\Results\control data'
sweeping_dir= 'D:\extSSD_backup\Glymphatic_BrainSweeping_Data\Results\Sweeping data'

# Define the 26 time points (Pre, 0.5h, 1h, 2h, ... 24h)
time_points = [f"{i:02d}. {label}" for i, label in enumerate(["Pre", "0.5h"] + [f"{j}h" for j in range(1, 25)])]
print(f"Time Points: {time_points}")

###########################################
# 8. Calculate Image Metrics using Metrics() function
# Loop over thresholds 0 to 5 and output the p-values.
print("\n=== Metrics with Varying Thresholds ===")
for t in range(0, 6):
    metrics_results = Metrics(control_dir=control_dir, sweeping_dir=sweeping_dir, threshold=t)
    print(f"\nThreshold = {t}")
    print(f"Control Time-to-Peak Indices: {metrics_results['control_time_to_peak_indices']}")
    print(f"Sweeping Time-to-Peak Indices: {metrics_results['sweeping_time_to_peak_indices']}")
    print(f"Time-to-Peak Comparison p-value: {metrics_results['pvalue_ttp']}")
    print(f"Control Peak-to-Baseline Indices: {metrics_results['control_peak_to_baseline_indices']}")
    print(f"Sweeping Peak-to-Baseline Indices: {metrics_results['sweeping_peak_to_baseline_indices']}")
    print(f"Peak-to-Baseline Comparison p-value: {metrics_results['pvalue_ptb']}")

###########################################
# New Metric: signal_variance
# Calculation: ((current timepoint signal ratio) - (previous timepoint signal ratio)) / (previous timepoint signal ratio) * 100
# For each subject, given 26 time points, 25 signal_variance values are obtained.
# Then, for each of the 25 time points, perform Wilcoxon ranksum test comparing sweeping vs. control.

import pandas as pd

def read_signal_variance(csv_files):
    """
    Reads the 'Signal Difference Ratio (%)' from each CSV file and calculates signal_variance.
    Returns an array of shape (n_subjects, 25) where each row corresponds to a subject.
    """
    variances = []
    for file in csv_files:
        df = pd.read_csv(file)
        sdr = df['Signal Difference Ratio (%)'].values  # Expecting 26 values
        if len(sdr) < 2:
            print(f"File {file} does not have enough time points.")
            continue
        # Calculate signal_variance for timepoints 1 to 25 using the previous timepoint as reference.
        variance = ((sdr[1:] - sdr[:-1]) / sdr[:-1]) * 100
        variances.append(variance)
    return np.array(variances)

# Get list of CSV files (only those starting with "Signal_Results") for control and sweeping groups.
control_files = [os.path.join(control_dir, f) for f in os.listdir(control_dir) if f.startswith('Signal_Results') and f.endswith('.csv')]
sweeping_files = [os.path.join(sweeping_dir, f) for f in os.listdir(sweeping_dir) if f.startswith('Signal_Results') and f.endswith('.csv')]

control_variances = read_signal_variance(control_files)
sweeping_variances = read_signal_variance(sweeping_files)

print("\n=== Signal Variance Wilcoxon Ranksum Test Results ===")
# Assuming each subject has 25 signal_variance values (from 26 time points)
if control_variances.shape[0] == 0 or sweeping_variances.shape[0] == 0:
    print("Not enough data for signal_variance analysis.")
else:
    for i in range(control_variances.shape[1]):  # for each of the 25 time points (starting from 0, which corresponds to time_points[1])
        stat, p_val = ranksums(control_variances[:, i], sweeping_variances[:, i])
        # For labeling, use the corresponding time point label (i+1) because variance is computed from previous to current.
        print(f"Time Point {time_points[i+1]} (signal variance): p-value = {p_val}")

# Compare sweeping vs. control 'signal_ratio' for each time point using Wilcoxon ranksum test
def read_signal_ratio(csv_files):
    """
    Reads the 'Signal Difference Ratio (%)' values from CSV files.
    Returns an array of shape (n_subjects, n_time_points).
    """
    signal_ratios = []
    for file in csv_files:
        with open(file, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            ratios = [float(row[2]) for row in reader]
            signal_ratios.append(ratios)
    return np.array(signal_ratios)

# Get list of CSV files (only those starting with "Signal_Results") for control and sweeping groups.
control_files = [os.path.join(control_dir, f) for f in os.listdir(control_dir)
                 if f.startswith('Signal_Results') and f.endswith('.csv')]
sweeping_files = [os.path.join(sweeping_dir, f) for f in os.listdir(sweeping_dir)
                  if f.startswith('Signal_Results') and f.endswith('.csv')]

control_ratios = read_signal_ratio(control_files)
sweeping_ratios = read_signal_ratio(sweeping_files)

print("\n=== Signal Ratio Wilcoxon Ranksum Test Results ===")
# Assuming each CSV contains 26 time points
for i in range(control_ratios.shape[1]):
    stat, p_val = ranksums(control_ratios[:, i], sweeping_ratios[:, i])
    print(f"Time Point {time_points[i]}: p-value = {p_val}")
