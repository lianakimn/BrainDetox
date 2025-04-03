import logging
import matplotlib
from ROIselection import roi_selection
from roipoly import RoiPoly

matplotlib.use('Qt5Agg')  # Ensure this matches your installed backend
matplotlib.use('MacOSX')  # Alternative backend

matplotlib.use('Qt5Agg')  # need this backend for ROIPoly to work
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import numpy as np
import SimpleITK as sitk
import os
import csv
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

############################################################################################################
###### Modify the following paths to match your directory structure ######
# Example: Assume if your dataset is saved in the following directory structure:
# C:\Users\MTNL\SNUH\Brain_Data\Control\Subject1\sub-01\00. Pre\...

control_dir = '/Users/lianakim/Downloads/Glymphatic_BrainSweeping_Data/Control/'
sweeping_dir= '/Users/lianakim/Downloads/Glymphatic_BrainSweeping_Data/Sweeping/'

# And Press the Run!!!

############################################################################################################
# 1. Load DICOM files (4D data in real-experimental scenario)

# Function to read DICOM files from a directory
def read_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_series)

    print(directory)

    image = reader.Execute()
    return image
def montage(image, title="Image"):
    array = sitk.GetArrayFromImage(image)
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        if i < array.shape[0]:
            ax.imshow(array[i, :, :], cmap="gray")
        ax.axis('off')
    fig.suptitle(title)
    plt.show()

def MRI_processing(main_dir, base_dir, time_points):
    # Initialize list to store 3D images
    images_3d = []
    # Loop through each time point directory
    input("@@@@@@@@@@@@@@@@     Press Enter to continue...   @@@@@@@@@@@@@@@@@@@@")
    for time_point in time_points:
        time_point_dir = os.path.join(base_dir, time_point)
        subfolders = sorted([f for f in os.listdir(time_point_dir) if not f.startswith('.')])

        
        # Assuming the T1-weighted images are located under subfolders[0]/1/
        if len(subfolders) == 0:
            print(f"No subfolders found in {time_point_dir}, use the previous time point directory.")
            time_point_dir = os.path.join(base_dir, prev_time_point)
            subfolders = sorted([f for f in os.listdir(time_point_dir) if not f.startswith('.')])
            t1_folder = os.path.join(time_point_dir, subfolders[0], '1')
        else:
            t1_folder = os.path.join(time_point_dir, subfolders[0], '1')
        
        # Read the 3D DICOM series
        image_3d = read_dicom_series(t1_folder)
        images_3d.append(image_3d)
        prev_time_point = time_point

    # Convert list of 3D images to a 4D image
    # Resample images to the same physical space
    reference_image = images_3d[0]
    resampled_images_3d = []
    for image in images_3d:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampled_image = resampler.Execute(image)
        resampled_images_3d.append(resampled_image)

    images_4d = sitk.JoinSeries(resampled_images_3d)
    image_data_4d = sitk.GetArrayFromImage(images_4d)
    print(f"@@@@ 4D Image Shape: {image_data_4d.shape}")

    ############################################################################################################
    # 2. 3D image registration of 4d data using simpleITK (motion correction)
    montage(sitk.GetImageFromArray(image_data_4d[0, :, :, :].astype(np.float32)), title="Fixed Image (pre-image)")
    montage(sitk.GetImageFromArray(image_data_4d[1, :, :, :].astype(np.float32)), title="Moving Image (30min post-image)")

    registered_4d_image = np.zeros_like(image_data_4d)
    registered_4d_image[0,:,:,:] = image_data_4d[0,:,:,:]

    registered_4d_image = image_data_4d.copy() # Skip the registration for now
    # for time_point in range(25):
    #     # Load the fixed and moving images (assuming 4D data where the last dimension is time)
    #     fixed_image = sitk.GetImageFromArray(image_data_4d[0, :, :, :].astype(np.float32))
    #     moving_image = sitk.GetImageFromArray(image_data_4d[time_point+1, :, :, :].astype(np.float32))
    #     if time_point == 0:
    #         montage(fixed_image, title="Fixed Image (pre-image)")
    #         montage(moving_image, title="Moving Image (30min post-image)")

    #     # Initialize the registration method
    #     registration_method = sitk.ImageRegistrationMethod()

    #     # Set the metric and optimizer
    #     registration_method.SetMetricAsMeanSquares()
    #     registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

    #     # Set the initial transform
    #     initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    #     registration_method.SetInitialTransform(initial_transform, inPlace=False)

    #     # Perform the registration
    #     final_transform = registration_method.Execute(fixed_image, moving_image)

    #     # Apply the final transform to the moving image
    #     resampler = sitk.ResampleImageFilter()
    #     resampler.SetReferenceImage(fixed_image)
    #     resampler.SetInterpolator(sitk.sitkLinear)
    #     resampler.SetDefaultPixelValue(0)
    #     resampler.SetTransform(final_transform)

    #     # Resample the moving image
    #     registered_image = resampler.Execute(moving_image)

    #     # Convert the registered image to a numpy array for visualization
    #     registered_image_array = sitk.GetArrayFromImage(registered_image)
    #     registered_4d_image[time_point+1, :, :, :] = registered_image_array


    ############################################################################################################
    # 3. Select target slice for analysis
    pre_image = registered_4d_image[0, :, :, :]
    post_image = registered_4d_image[1, :, :, :]

    # Function to display slices interactively with arrow keys
    def show_slices_with_arrows(pre_image):
        """
        Display slices of a 3D MRI volume (pre_image) interactively.
        Press the right arrow key (->) to go forward one slice.
        Press the left arrow key (<-) to go back one slice.
        """
        # Initialize current slice index
        current_slice = [0]  # Use a list for mutability within the event function
        
        fig, ax = plt.subplots()
        
        # Initial display
        ax.imshow(pre_image[current_slice[0], :, :], cmap="gray")
        ax.set_title(f"Slice # {current_slice[0]+1} / {pre_image.shape[0]}")
        
        # Function to update the displayed slice
        def update_slice(index):
            ax.clear()
            ax.imshow(pre_image[index,:,:], cmap="gray")
            ax.set_title(f"Slice # {index+1} / {pre_image.shape[0]}")
            fig.canvas.draw()
        
        # Event handler for key presses
        def on_key(event):
            if event.key == 'right':  # Next slice
                current_slice[0] = min(current_slice[0] + 1, pre_image.shape[0] - 1)
                update_slice(current_slice[0])
            elif event.key == 'left':  # Previous slice
                current_slice[0] = max(current_slice[0] - 1, 0)
                update_slice(current_slice[0])
        
        # Connect the event handler to the figure
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()

    # interactively dispaly post_image slice-by-slice
    show_slices_with_arrows(post_image)

    # get user keyboard number typing input and save into parameter 'target_slice'
    target_slice = int(input(f"Enter the slice number for ROI selection (1-{post_image.shape[0]}): "))-1

    # Display the registered image
    normalized_slice_sitk = sitk.GetImageFromArray(registered_4d_image[:, target_slice, :, :].astype(np.float32))
    normalized_slice = sitk.GetArrayFromImage(normalized_slice_sitk)  # Now it's a NumPy array
    montage(normalized_slice_sitk, title=f"Selected slice number: {target_slice+1}")
    ############################################################################################################

    # 4. Apply Gaussian filter for noise reduction (optional)

    # target_slice = 50
    pixel_array = registered_4d_image[:, target_slice, :, :]  # Select a single slice across all time points
    smoothed_image = gaussian_filter(pixel_array, sigma=0.1)

    # Display smoothed image for the first time point
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the original image
    axs[0].imshow(pixel_array[10, :, :], cmap="gray")
    axs[0].set_title(f"Before Denoising - Slice #{target_slice+1}")
    axs[0].axis("off")

    # Display the denoised image
    axs[1].imshow(smoothed_image[10, :, :], cmap="gray")
    axs[1].set_title(f"After Denoising - Slice #{target_slice+1}")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    ############################################################################################################
    # 5. Set ROI (Need 2 ROIs, 1 for normalization and 1 for the glymphatic signal region of interest)

    logging.basicConfig(
        format='%(levelname)s %(processName)-10s : %(asctime)s '
               '%(module)s.%(funcName)s:%(lineno)s %(message)s',
        level=logging.INFO )

    logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                               '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                        level=logging.INFO)

    averaged_image = np.mean(smoothed_image, axis=0)  # 20250224: Average across all time points for visualization

    # Show the image for the first time point and let user draw first ROI
    fig = plt.figure()
    plt.imshow(averaged_image, cmap="gray")  # 20250224: Average across all time points for visualization
    plt.colorbar()
    plt.title("PREPARE brain ROI; left click: line segment; right click or double click: close region")
    plt.show(block=True)  # Ensure this blocks execution

    # Let user draw first ROI
    selected_roi1, roi1 = roi_selection(averaged_image)  # 20250224: Average across all time points for visualization
    print("ROI1 selected successfully!")

    # Confirm ROI creation before proceeding
    print("ROI1 creation confirmed!")
    print("Selected Points:", selected_roi1)  # Debugging the coordinates
    print("Mask Shape:", roi1.shape)  # Should match the image shape
    print("Mask Unique Values:", np.unique(roi1))  # Should show [0, 1]

    # Show the image with the first ROI mask overlaid
    fig = plt.figure()
    plt.imshow(averaged_image, cmap="gray")  # 20250224: Average across all time points for visualization
    plt.colorbar()
    plt.contour(roi1, colors="r")  # Overlay the ROI mask as a contour in red
    plt.title('ROI for glymphatic region')
    plt.show(block=False)
    plt.imshow(roi1, cmap="gray")
    plt.title("Binary ROI Mask")
    plt.show()

    # Let user draw second ROI
    fig = plt.figure()
    plt.imshow(averaged_image, cmap="gray")  # 20250224: Average across all time points for visualization
    plt.colorbar()
    plt.title("PREPARE muscle ROI; left click: line segment; right click or double click: close region")
    plt.show(block=True)  # Ensure this blocks execution

    selected_roi2, roi2 = roi_selection(averaged_image)  # 20250224: Average across all time points for visualization
    print("ROI2 selected successfully!")

    # Confirm ROI2 creation
    print("ROI2 creation confirmed!")
    print("Selected Points for ROI2:", selected_roi2)
    print("ROI2 Mask Shape:", roi2.shape)
    print("ROI2 Mask Unique Values:", np.unique(roi2))

    # Display the binary mask and contour for ROI2
    fig = plt.figure()
    plt.imshow(averaged_image, cmap="gray")  # 20250224: Average across all time points for visualization
    plt.colorbar()
    plt.contour(roi2, colors="b")  # Overlay the ROI2 mask as a contour in blue
    plt.title("ROI for glymphatic region (ROI2)")
    plt.show()

    plt.imshow(roi2, cmap="gray")
    plt.title("Binary ROI Mask (ROI2)")
    plt.show()

    # Overlay ROI1 and ROI2 on the original image
    plt.imshow(averaged_image, cmap="gray")  # 20250224: Average across all time points for visualization
    plt.contour(roi1, colors="r", label="ROI1 (Brain Region)")  # Red: Brain region
    plt.contour(roi2, colors="b", label="ROI2 (Muscle Region)")  # Blue: Muscle region
    plt.title("ROIs: Brain (Glymphatic Region) and Muscle (Reference Region)")
    plt.legend(["ROI1 (Brain)", "ROI2 (Muscle)"])
    plt.show()

    # Show the combined ROI masks
    combined_mask = roi1 + roi2  # Combine both binary masks
    plt.imshow(combined_mask, cmap="gray")
    plt.title("Combined ROI Masks")
    plt.show()

    ############################################################################################################
    # 6. Image Normalization (based on the muscle ROI signal)

    # Extract muscle signal using the binary mask (roi2 should be boolean with shape (256, 192))
    muscle_signal = smoothed_image[:, roi2]  # Shape: (26, number of selected pixels)

    # Calculate the mean signal intensity of the muscle ROI across all time points
    mean_muscle_signal = np.mean(muscle_signal, axis=1, keepdims=True)  # Shape: (26, 1)
    mean_muscle_signal = mean_muscle_signal / mean_muscle_signal[
        0]  # 20250224. muscle signal normalization prepapration

    print(f"Mean Signal Intensity in Muscle ROI: {mean_muscle_signal.shape}")  # Debugging
    print(mean_muscle_signal)

    ############################################################################################################
    # 7. Extract the glymphatic ROI signal from the normalized smoothed image

    print(f"Shape of normalized_slice: {normalized_slice.shape}")  # Should be (time, height, width)
    print(f"Shape of roi_mask: {roi1.shape}")  # Should be (height, width)

    roi_mask = roi1  # Since roi1 is already a binary mask
    roi_signal = smoothed_image[:, roi_mask.astype(bool)]  # 20250224. Please check.
    print(roi_signal.shape)
    # Analyze signal intensity across all time points
    mean_signal_raw = np.mean(roi_signal, axis=1)  # 20250224. Please check.
    print(f"Mean Signal Intensity in brain ROI across all time points: {mean_signal_raw}")

    # Plot histogram of pixel intensities for the first time point

    # Plot the mean signal intensity across all time points
    plt.plot(mean_signal_raw, marker='o', linestyle='-', color='b')
    plt.title("Mean Signal Intensity in brain ROI Across All Time Points")
    plt.xlabel("Time Point")
    plt.ylabel("Mean Intensity")
    plt.grid(True)
    plt.show()

    ############################################################################################################
    # 8. Calculate some Image metrics

    # normalized mean_signal_raw using the mean_muscle_signal
    mean_signal = mean_signal_raw / mean_muscle_signal.flatten()  # 20250224. Muscle signal normalization
    print(
        f"Shape of mean_signal_raw: {mean_signal_raw.shape}")  # 20250224. Only for debugging, you can remove this line.
    print(
        f"Shape of mean_muscle_signal: {mean_muscle_signal.shape}")  # 20250224. Only for debugging, you can remove this line.
    print(f"Shape of mean_signal: {mean_signal.shape}")  # 20250224. Only for debugging, you can remove this line.

    # Signal difference ratio
    baseline_signal = mean_signal[0] #preimage signal = baseline signal
    signal_difference_ratio = (mean_signal - baseline_signal) / baseline_signal * 100

    # Time-to-peak

    time_to_peak_index = np.argmax(mean_signal)
    print(f"time_to_peak_index: {time_to_peak_index}")
    print(f"Length of time_points: {len(time_points)}")

    time_to_peak = time_points[time_to_peak_index]

    ################################################################ 20250224. Starting Peak-to-baseline calculation
    # Peak-to-baseline: find the first time point after the peak where the signal is below the baseline.
    post_peak_signal = mean_signal[time_to_peak_index:]
    if np.any(post_peak_signal < baseline_signal):
        # Find the first index (relative to post_peak_signal) where the value is below the baseline.
        relative_index = np.argmax(post_peak_signal < baseline_signal)
    else:
        # If no values are below the baseline, use the index of the minimum value in post_peak_signal.
        relative_index = np.argmin(post_peak_signal)

    # Calculate the absolute index in time_points.
    peak_to_baseline_index = time_to_peak_index + relative_index
    peak_to_baseline = time_points[peak_to_baseline_index]
    ################################################################ 20250224. End of Peak-to-baseline calculation
    # Print the calculated metrics
    print(f"Time-to-Peak: {time_to_peak}")
    print(f"Peak-to-Baseline: {peak_to_baseline}")

    # Plot the signal difference ratio
    plt.plot((signal_difference_ratio), marker='o', linestyle='-', color='r')
    plt.title("Signal Difference Ratio Across All Time Points")
    plt.xlabel("Time Pointime_points, t")
    plt.ylabel("Signal Difference Ratio (%)")
    plt.grid(True)
    plt.show()

    ############################################################################################################
    # 9. Save the results to a CSV file
    # Define the CSV file paths
    subject_id = os.path.basename(base_dir)
    csv_file_path = os.path.join(main_dir, f"Signal_Results_slicenumber_{target_slice}_{subject_id}.csv")
    csv_file_path_2 = os.path.join(main_dir, f"Time_Peak_Baseline_Results_{target_slice}{subject_id}.csv")

    # Write the results to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time Point", "Mean Signal Intensity", "Signal Difference Ratio (%)"])
        
        for i, time_point in enumerate(time_points):
            writer.writerow([time_point, mean_signal[i], signal_difference_ratio[i]])

    print(f"Results saved to {csv_file_path}")

    # CSV file to save time-to-peak and peak-to-baseline
    with open(csv_file_path_2, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time-to-Peak", "Peak-to-Baseline"])
        writer.writerow([time_to_peak, peak_to_baseline])
        writer.writerow(["Time-to-Peak-index", "Peak-to-Baseline-index"])
        writer.writerow([time_to_peak_index, peak_to_baseline_index])

    print(f"Results saved to {csv_file_path_2}")


############################################################################################################
# --- Run the MRI processing pipeline ---
runFlag = 1
if runFlag==1:
    # Ensure the runFlag = 1
    # if runFlag==1:necessary directories exist
    if not os.path.exists(control_dir):
        raise FileNotFoundError("The specified directory does not exist.")
    os.chdir(control_dir)

    # Directories
    time_points = [f"{i:02d}. {label}" for i, label in enumerate(["Pre", "0.5h"] + [f"{j}h" for j in range(1, 25)])]
    print(f"Time Points: {time_points}")

    # Define the main directories
    main_dirs = [control_dir]
    #change control_dir to sweeping_dir to select

    # Iterate through each main directory
    for main_dir in main_dirs:
        # Get the list of subjects
        subjects = sorted(os.listdir(main_dir))
        print(f"Found {len(subjects)} subjects in {main_dir}")
        print(f"Subjects: {subjects}")

        # Assign numbers to each subject
        subject_dict = {i + 1: subject for i, subject in enumerate(subjects)}

        # Display the list of subjects with their assigned numbers
        print("Available subjects:")
        for num, subject in subject_dict.items():
            print(f"{num}: {subject}")

        # Ask the user for a starting subject number
        start_num = input("Enter the starting subject number (or press Enter to process all): ").strip()

        # Convert input to integer if provided
        start_num = int(start_num) if start_num.isdigit() else None

        # Process each subject, starting from the chosen number
        for num, subject in subject_dict.items():
            if not subject.startswith('.'):  # Ignore hidden files

                # Skip until we reach the selected number
                if start_num and num < start_num:
                    continue

                base_dir = os.path.join(main_dir, subject)
                MRI_processing(main_dir, base_dir, time_points)

                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(f"Processing {subject}")

############################################################################################################
statFlag = 0
if statFlag == 1:
    # --- Statistical Analysis ---
    # Load the CSV files for control and sweeping groups
    control_files = [os.path.join(control_dir, f) for f in os.listdir(control_dir) if f.startswith('Signal_Results')]
    sweeping_files = [os.path.join(sweeping_dir, f) for f in os.listdir(sweeping_dir) if f.startswith('Signal_Results')]
    time_points = [f"{i:02d}. {label}" for i, label in enumerate(["Pre", "0.5h"] + [f"{j}h" for j in range(1, 25)])]

    # Function to read signal difference ratios from CSV files
    def read_signal_difference_ratios(csv_files):
        signal_difference_ratios = []
        for file in csv_files:
            with open(file, mode='r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                ratios = [float(row[2]) for row in reader]
                signal_difference_ratios.append(ratios)
        return np.array(signal_difference_ratios)

    # Read the signal difference ratios
    control_ratios = read_signal_difference_ratios(control_files)
    sweeping_ratios = read_signal_difference_ratios(sweeping_files)

    # Perform t-tests at each time point
    p_values = []
    for i in range(control_ratios.shape[1]):
        t_stat, p_val = ttest_ind(control_ratios[:, i], sweeping_ratios[:, i])
        p_values.append(p_val)

    # Print the p-values for each time point
    for i, p_val in enumerate(p_values):
        print(f"Time Point {time_points[i]}: p-value = {p_val}")

    # Load the CSV files for time-to-peak and peak-to-baseline metrics
    control_metrics_files = [os.path.join(control_dir, f) for f in os.listdir(control_dir) if f.startswith('_Time_Peak_Baseline_Results.csv')]
    sweeping_metrics_files = [os.path.join(sweeping_dir, f) for f in os.listdir(sweeping_dir) if f.startswith('_Time_Peak_Baseline_Results.csv')]

    # Function to read time-to-peak and peak-to-baseline metrics from CSV files
    def read_metrics(csv_files):
        time_to_peak = []
        peak_to_baseline = []
        for file in csv_files:
            with open(file, mode='r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                metrics = next(reader)
                time_to_peak.append(metrics[0])
                peak_to_baseline.append(metrics[1])
        return time_to_peak, peak_to_baseline

    # Read the metrics
    control_time_to_peak, control_peak_to_baseline = read_metrics(control_metrics_files)
    sweeping_time_to_peak, sweeping_peak_to_baseline = read_metrics(sweeping_metrics_files)

    # Convert metrics to numerical values for statistical testing
    control_time_to_peak = np.array([time_points.index(tp) for tp in control_time_to_peak if tp != "N/A"])
    sweeping_time_to_peak = np.array([time_points.index(tp) for tp in sweeping_time_to_peak if tp != "N/A"])

    # Perform t-tests for time-to-peak
    t_stat, p_val = ttest_ind(control_time_to_peak, sweeping_time_to_peak)
    print(f"Time-to-Peak: p-value = {p_val}")

    # Note: Peak-to-baseline may contain "N/A" values, handle accordingly
    control_peak_to_baseline = np.array([time_points.index(pb) for pb in control_peak_to_baseline if pb != "N/A"])
    sweeping_peak_to_baseline = np.array([time_points.index(pb) for pb in sweeping_peak_to_baseline if pb != "N/A"])

    # Perform t-tests for peak-to-baseline
    t_stat, p_val = ttest_ind(control_peak_to_baseline, sweeping_peak_to_baseline)
    print(f"Peak-to-Baseline: p-value = {p_val}")



############################################################################################################
# # Optional

# # Convert DICOM to NIfTI
# # Read DICOM series
# reader = sitk.ImageSeriesReader()
# dicom_series = reader.GetGDCMSeriesFileNames("-----\MRI_sample_data\subject1_Sagittal_3D_T1wighted_imaging")
# reader.SetFileNames(dicom_series)
# image = reader.Execute()
# os.chdir("-----\MRI_sample_data\subject1_Sagittal_3D_T1wighted_imaging")

# # Save as NIfTI
# sitk.WriteImage(image, "output_image.nii.gz")

# # 3D Image Visualization
# nifti_image = nib.load("output_image.nii.gz")
# image_data = nifti_image.get_fdata()

# # Display a single slice
# plt.imshow(image_data[:, :, 50], cmap="gray")
# plt.title("Slice 50 of 3D MRI")
# plt.show()
