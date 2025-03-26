import os
import re
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.signal import iirnotch, filtfilt, lfilter
from spectral_connectivity import Multitaper, Connectivity




  
    
def split_by_phase(df, TT_to_drop):
    
    df = df.drop(columns=TT_to_drop)
    sample_df = df.loc[df['phase']=='Sample'].drop(['phase'], axis=1)
    delay_df = df.loc[df['phase']=='Delay'].drop(['phase'], axis=1)
    test_df = df.loc[df['phase']=='Test'].drop(['phase'], axis=1)
    iti_df = df.loc[df['phase']=='ITI'].drop(['phase'], axis=1)
    
    return sample_df, delay_df, test_df, iti_df



    
def prepare_for_multitaper(df):
    '''
    Prepare for multitaper by reshaping dataframe
    df: DataFrame.
    '''
    
    print("Original DataFrame shape:", df.shape)   
    # Drop unnecessary columns
    to_drop = ['start_time', 'end_time', 'timestamp']
    df = df.drop(to_drop, axis=1)
    
    # Check for NaN values in each column
    nan_summary = df.isna().sum()
    print("Missing Data Summary Before Reshaping:\n", nan_summary)
    
    # Create TT list
    tts = [col for col in df.columns if col.startswith('TT')]
     
    # Re-shape dataset into array
    reshaped = df.pivot_table(
        index='i',
        columns='ripple_nr',
        values=tts,
        dropna=False
    ).values
    
    reshaped = reshaped.reshape(
        (
            df['i'].nunique(),
            df['ripple_nr'].nunique(),
            len(tts)
        )
    )
    print("Final reshaped array shape:", reshaped.shape)      
    print("Nans in reshaped:", np.isnan(reshaped).sum())
    reshaped[np.isnan(reshaped)] = np.nanmean(reshaped)
    return reshaped
    
    
def apply_notch_filter_3d(data, notch_freq, fs, quality_factor):
    """
    Apply a notch filter to a 3D numpy array with time series data.
    Parameters:
    - data: 3D numpy array with shape (number_of_samples, trials, tetrodes)
    - notch_freq: Frequency to notch out (in Hz)
    - fs: Sampling frequency (in Hz)
    - quality_factor: Quality factor of the notch filter
    Returns:
    - filtered_data: 3D numpy array with the notch filter applied
    """
    # Ensure the input data is a numpy array
    data = np.array(data)
    
    # Check for NaNs in the input data
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaNs. Please clean data before filtering.")    
    # Design the notch filter
    nyquist = 0.5 * fs
    normal_cutoff = notch_freq / nyquist
    b, a = iirnotch(normal_cutoff, quality_factor)   
    # Initialize the filtered_data array
    filtered_data = np.empty_like(data)    
    # Iterate over each time sample
    for j in range(data.shape[1]):  # Iterate over trials
        for k in range(data.shape[2]):  # Iterate over tetrodes
                
            # Extract the 1D array for filtering
            signal = data[:, j, k]              
            # Apply the notch filter if signal is valid
            if np.size(signal) > 1:       
                filtered_data[:, j, k] = lfilter(b, a, signal)

    return filtered_data  
    
    
def get_conn (df, fs, step, window, exp_type, nblocks, time_halfbandwidth_product):
    
    '''
    Prepares dataframe provided for multitaper
    Creates a multitaper object
    Inputs the multitaper object to create a connectivity object.
    '''
    
    multitaper = Multitaper(
        df, 
        sampling_frequency=fs, 
        n_time_samples_per_window=window,
        n_time_samples_per_step = step,
        time_halfbandwidth_product = time_halfbandwidth_product
    )
    
    print(multitaper)            
    conn = Connectivity.from_multitaper(
        multitaper, 
        expectation_type = exp_type, 
        blocks = nblocks 
    )
    
    return conn   
    
def get_metrics(conn):
    
    """
    Calculate the average power and standard deviation across tetrodes and trials.  
    Parameters:
    conn : connectivity object
        An object containing LFP data, with methods to access power, time, and frequency information.  
    Returns:
        freqs (np.ndarray): Array of frequency values.
        times (np.ndarray): Array of time labels.
        power (np.ndarray): Original power array with dimensions (Time windows, trials, Frequencies, Tetrodes).
    """

    # Retrieve time labels and frequencies from the connectivity object
    times = conn.time
    freqs = conn.frequencies  
    # Calculate power with dimensions (Time windows, trials, Frequencies, Tetrodes)
    power = conn.power()     
    return freqs, times, power

def save_arrays(array1, array2, array3, save_dir, filename1, filename2, filename3):
    """
    Save three arrays to disk in a specified directory, where the first two are 2D arrays and the third is a 4D array.
    Parameters:
    - array1: First 2D numpy array
    - array2: Second 2D numpy array
    - array3: Third 4D numpy array
    - save_dir: Directory path where arrays will be saved
    - filename1: Filename for saving the first 2D array 
    - filename2: Filename for saving the second 2D array
    - filename3: Filename for saving the third 4D array
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)  
    # Save arrays to .npy files
    np.save(os.path.join(save_dir, filename1), array1)
    np.save(os.path.join(save_dir, filename2), array2)
    np.save(os.path.join(save_dir, filename3), array3)
    
    
    
def load_arrays(path, filename1, filename2, filename3):
    """
    Load three arrays from disk in a specified directory, where the first two are 2D arrays and the third is a 4D array.
    Parameters:
    - path: Directory path where arrays are saved
    - filename1: Filename for the first 2D array
    - filename2: Filename for the second 2D array
    - filename3: Filename for the third 4D array
    Returns:
    - array1: Loaded 2D numpy array from filename1
    - array2: Loaded 2D numpy array from filename2
    - array3: Loaded 4D numpy array from filename3
    """

    # Load arrays from .npy files
    array1 = np.load(os.path.join(path, filename1))
    array2 = np.load(os.path.join(path, filename2))
    array3 = np.load(os.path.join(path, filename3))
    return array1, array2, array3



def zscore_4d_power(data):
    """
    Z-score the power values in a 4D array by each sample.
    Parameters:
    - data: 4D numpy array of shape (time_bins, trials, freq_bins, samples/tetrodes)
    Returns:
    - 4D numpy array with z-scored power values, maintaining the same shape as the input data
    """
    # Calculate mean and std withn trial and tetrode (across time_bins and freq_bins)
    mean = np.mean(data, axis=(0, 2), keepdims=True)
    std = np.std(data, axis=(0, 2), keepdims=True)   
    return (data - mean) / std  



def remove_outliers_zscored(data, z_threshold):
    """
    Detect and remove outliers from a z-scored 4D array by substituting outliers with the average
    within their respective trial and tetrode. Provides a comparative visualization of the power 
    distributions before and after outlier removal.
    Parameters:
    - data: 4D numpy array of z-scored power values with shape (time, trials, frequencies, tetrodes)
    - z_threshold: The z-score threshold for detecting outliers
    Returns:
    - cleaned_data: 4D numpy array with outliers removed, maintaining the same shape
    - percentage_removed: Percentage of data points removed as outliers
    """
    
    # Copy data to avoid modifying the original
    data_cleaned = np.copy(data)   
    # Calculate the percentage of outliers and replace them within their respective trial and tetrode
    total_elements = np.prod(data.shape)
    outliers_count = 0

    for trial in range(data.shape[1]):
        for tetrode in range(data.shape[3]):
            # Isolate the data slice for the current trial and tetrode
            slice_data = data[:, trial, :, tetrode]

            # Identify outliers in this slice
            outliers = np.abs(slice_data) > z_threshold

            # Count the outliers
            outliers_count += np.sum(outliers)

            # Compute the mean of the non-outlier values in this slice
            slice_mean = np.nanmean(slice_data[~outliers])

            # Replace outliers with the slice mean
            slice_data[outliers] = slice_mean

            # Assign the cleaned slice back to the cleaned_data array
            data_cleaned[:, trial, :, tetrode] = slice_data

    # Calculate the percentage of data points removed as outliers
    percentage_removed = (outliers_count / total_elements) * 100
    print(f"Percentage of data points replaced as outliers: {percentage_removed:.2f}%")
    # Comparative visualization of power distributions
    
    plt.figure(figsize=(12, 6))  
    # Before removal
    plt.subplot(1, 2, 1)
    sns.histplot(data.flatten(), bins=50, kde=True, color='blue')
    plt.title('Power Distribution Before Outlier Removal')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')
    # After removal
    plt.subplot(1, 2, 2)
    sns.histplot(data_cleaned.flatten(), bins=50, kde=True, color='green')
    plt.title('Power Distribution After Outlier Removal')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')  
    plt.tight_layout()
    plt.show()

    return data_cleaned

def average_power(data):
    """
    Calculate the average power and standard deviation.
    Parameters:
    - data: 4D numpy array with shape (time_bins, trials, freq_bins, samples)
    Returns:
    - avg_power: 2D numpy array of shape (time_bins, freq_bins) with average power
    - std_power: 2D numpy array of shape (time_bins, freq_bins) with standard deviation of power
    """
    # Calculate average power across tetrodes (axis3) and trials (axis1)
    avg_power = np.mean(data, axis=(3, 1)) 
    # Calculate standard deviation of power across tetrodes (axis3) and trials (axis1)
    std_power = np.std(data, axis=(3,1)) 
    # Calculate median power
    median_power = np.median(data, axis=(3,1))
    return avg_power, std_power, median_power

def plot_psd_band(time_bins, frequency_bins, power, x_tick_labels=None, ylim=None, vmin=None, vmax=None):
    """
    Create a Power Spectral Density (PSD) plot using imshow with Gaussian interpolation.

    Parameters:
    - time_bins: 1D numpy array of time bins
    - frequency_bins: 1D numpy array of frequency bins
    - power: 2D numpy array of power values with shape (time_bins, frequency_bins)
    - x_tick_labels: List or array of custom labels for the x-axis (time bins)
    - ylim: Tuple specifying the y-axis limits (min, max) in frequency_bins or None to auto-scale
    - vmin: Minimum value for the color scale or None to auto-scale based on filtered data
    - vmax: Maximum value for the color scale or None to auto-scale based on filtered data
    """
    # Ensure inputs are correctly shaped
    if power.shape[0] != len(time_bins) or power.shape[1] != len(frequency_bins):
        raise ValueError("Shape of power array must match the length of time_bins and frequency_bins.")

    # Filter time_bins and corresponding power values to only include those between 0 and 1
    valid_indices = np.where((time_bins >= 0) & (time_bins <= 1))[0]
    filtered_time_bins = time_bins[valid_indices]
    filtered_power = power[valid_indices]

    # Filter the power and frequency_bins based on ylim
    ylim_indices = [np.searchsorted(frequency_bins, val) for val in ylim]
    filtered_frequency_bins = frequency_bins[ylim_indices[0]:ylim_indices[1]]
    filtered_power = filtered_power[:, ylim_indices[0]:ylim_indices[1]]

    # Recalculate vmin and vmax based on the filtered power data if not provided
    if vmin is None:
        vmin = np.min(filtered_power)
    if vmax is None:
        vmax = np.max(filtered_power)

    # Set up the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4), dpi=300)

    # Plot using imshow
    plt.imshow(
        filtered_power.T,  # Transpose to match the orientation (time on x-axis, frequency on y-axis)
        aspect='auto',  # Automatic aspect ratio
        cmap='jet',  # Color map
        origin='lower',  # Set origin to lower-left
        interpolation='gaussian',  # Use Gaussian interpolation
        vmin=vmin,  # Set the minimum value for the color scale
        vmax=vmax   # Set the maximum value for the color scale
    )

    # Add colorbar and labels
    plt.colorbar(label='Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Set y-ticks at every 5th tick, starting from 0 (for the filtered frequency bins)
    y_ticks = np.arange(0, len(filtered_frequency_bins), 5)   
    y_labels = (np.round(filtered_frequency_bins[::5], 2)).astype(int)  # Label every 5th frequency
    plt.yticks(ticks=y_ticks, labels=y_labels)

    # Set x-ticks using the provided x_tick_labels if available, rounded to one decimal place
    if x_tick_labels is not None:
        rounded_labels = [round(label, 2) for label in x_tick_labels]  # Round to one decimal place
        plt.xticks(ticks=np.arange(len(rounded_labels))-.5, labels=rounded_labels)
        print(np.arange(len(rounded_labels)))
        
    # Add vertical white dotted line where time_bin == 0.5
    #if 0.4 in filtered_time_bins:
        time_bin_index = np.searchsorted(filtered_time_bins, 0.5)-.5  # Find index where time_bin == 0.5
        plt.axvline(x=time_bin_index, color='white', linestyle='--', linewidth=2)
        
        print(time_bin_index)
        
    # Add title and show the plot
    #plt.title('Power Spectral Density (PSD)')
    plt.grid(False)
    sns.despine()


    print('Filtered Frequency bins: {}'.format(filtered_frequency_bins))
    print('Time bins: {}'.format(filtered_time_bins))
    
    
    
    
def convert_to_long_format(
    data_array, time_bin_values, frequency_bin_values, ripple_labels, tetrode_labels,
    time_bin_label='time_bins', ripple_label='ripple_nr', freq_bin_label='frequency_bins', 
    tetrode_label='tetrode', power_label='power'
    ):
    """
    Convert a 4D array (time_bins, trials, frequency_bins, tetrodes) into a long-format pandas DataFrame.   
    Parameters:
    - data_array: 4D numpy array with shape (time_bins, trials, frequency_bins, tetrodes)
    - time_bin_values: 1D numpy array or list of time bin labels
    - frequency_bin_values: 1D numpy array or list of frequency bin labels
    - ripple_labels: An array with the ripple numbers 
    - tetrode_labels: An array with tetrode numbers
    - time_bin_label: Column name for time bins (default: 'time_bins')
    - trial_label: Column name for trials (default: 'trials')
    - freq_bin_label: Column name for frequency bins (default: 'frequency_bins')
    - tetrode_label: Column name for tetrodes (default: 'tetrode')
    - power_label: Column name for power values (default: 'power')
    
    Returns:
    - long_format_df: A pandas DataFrame in long format with columns [time_bins, trials, frequency_bins, tetrode, power]
    """
    # Get the shape of the input array
    time_bins = data_array.shape[0]
    ripples = data_array.shape[1]
    frequency_bins = data_array.shape[2]
    tetrodes = data_array.shape[3]
    
    # Check that time_bin_values and frequency_bin_values match the shape of data_array
    if len(time_bin_values) != time_bins:
        raise ValueError(f"Length of time_bin_values ({len(time_bin_values)}) does not match number of time bins ({time_bins})")
    if len(frequency_bin_values) != frequency_bins:
        raise ValueError(f"Length of frequency_bin_values ({len(frequency_bin_values)}) does not match number of frequency bins ({frequency_bins})")
    
    # Reshape the 4D array into a 1D array (flatten the array)
    reshaped_power = data_array.reshape(time_bins * ripples * frequency_bins * tetrodes)
    
    # Create multi-index for each combination of time_bins, trials, frequency_bins, and tetrodes
    multi_index = pd.MultiIndex.from_product(
        [time_bin_values, ripple_labels, frequency_bin_values, tetrode_labels],
        names=[time_bin_label, ripple_label, freq_bin_label, tetrode_label]
    )
    
    # Create the dataframe with 'power' values and the multi-index
    long_format_df = pd.DataFrame({power_label: reshaped_power}, index=multi_index).reset_index() 
   
    return long_format_df






def plot_average_power(df, band=(4, 12), title=None, color='black', ylim=None, xlim=None):
    """
    Plot the power values with individual samples and the average overlay.

    Parameters:
    - df: DataFrame in long format with columns [time_bins, frequency_bins, power, ...]
    - band: Tuple to set the band to filter by
    - title: Title of the plot (default is band name)
    - color: Color of the average line (default is black)
    - ylim: Tuple to set the y-axis limits (e.g., (lower, upper))
    - xlim: Tuple to set the x-axis limits (e.g., (lower, upper))
    """
    # Set up the figure and axes
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set(context='talk', style='white')

    # Filter the DataFrame by the specified frequency band
    df_filtered = df[df.frequency_bins.between(band[0], band[1])]

    # Plot the average line with standard deviation shading
    sns.lineplot(
        data=df_filtered,
        x='time_bins',
        y='power',
        color=color,
        legend=False,
        errorbar='sd',  # Shows standard deviation as shaded area
        estimator=np.mean
    )

    # Add a vertical dashed line at time_bins == 0
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    # Set x and y limits if provided
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)

    # Add labels and title
    plt.xlabel('Time Bins')
    plt.ylabel('Power (z-score)')

    # Set title (either provided or default to band)
    plt.title(title if title else f'Power in {band[0]}-{band[1]} Hz Band')

    # Clean up the plot aesthetic
    sns.despine()
    plt.grid(False)
    plt.show()


def plot_median_power(df, band=(4, 12), title=None, color='black', ylim=None, xlim=None):
    """
    Plot the power values with the median and IQR overlay.

    Parameters:
    - df: DataFrame in long format with columns [time_bins, frequency_bins, power, ...]
    - band: Tuple to set the band to filter by
    - title: Title of the plot (default is band name)
    - color: Color of the median line (default is black)
    - ylim: Tuple to set the y-axis limits (e.g., (lower, upper))
    - xlim: Tuple to set the x-axis limits (e.g., (lower, upper))
    """
    # Set up the figure and axes
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set(context='talk', style='white')

    # Filter the DataFrame by the specified frequency band
    df_filtered = df[df.frequency_bins.between(band[0], band[1])]

    # Calculate the median and IQR for each time bin
    iqr_data = df_filtered.groupby('time_bins')['power'].agg(['median', lambda x: np.percentile(x, 75) - np.percentile(x, 25)])
    iqr_data.columns = ['median', 'iqr']
    
    # Calculate lower and upper bounds of the IQR
    iqr_data['iqr_lower'] = iqr_data['median'] - iqr_data['iqr'] / 2
    iqr_data['iqr_upper'] = iqr_data['median'] + iqr_data['iqr'] / 2

    # Plot the median line
    plt.plot(iqr_data.index, iqr_data['median'], color=color, label='Median')

    # Plot the IQR shaded area
    plt.fill_between(iqr_data.index, iqr_data['iqr_lower'], iqr_data['iqr_upper'], color=color, alpha=0.3, label='IQR')

    # Add a vertical dashed line at time_bins == 0
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    # Set x and y limits if provided
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)

    # Add labels and title
    plt.xlabel('Time Bins')
    plt.ylabel('Power (z-score)')

    # Set title (either provided or default to band)
    plt.title(title if title else f'Power in {band[0]}-{band[1]} Hz Band')

    # Clean up the plot aesthetic
    sns.despine()
    plt.grid(False)
    plt.legend(frameon=False)
    plt.show()

    

def analyze_specific_power_change(df, timepoint_pair, band=None, color='blue', bins=(0, 1, 0.1), xlim=None):
    """
    Analyze the distribution of paired power value changes between specified timepoints,
    and create a step histogram for the comparison.
    
    Arguments:
    df - The input dataframe with columns ['timepoint', 'power', 'frequency_bins'].
    timepoint_pair - Tuple of two timepoint labels to compare (e.g., ('Pre', 'Post')).
    band - Tuple to filter the dataframe by frequency_bins (e.g., (4, 8) for theta band), optional.
    color - Color for the step histogram line (default is 'blue').
    bins - Tuple specifying the range for the histogram bins (start, stop, step).
    xlim - Tuple specifying the limits for the x-axis (e.g., (min, max)), optional.    
    
    Returns:
    results - A dictionary with the average and median change, and the array of changes.
    """
    # Set the plot context and style
    sns.set(context='talk', style='white')
    
    # Validate timepoints in the dataframe
    tp1, tp2 = timepoint_pair
    if tp1 not in df['time_points'].unique() or tp2 not in df['time_points'].unique():
        raise ValueError("Both timepoints must be present in the dataframe.")
    
    # Filter dataframe by the given frequency band if provided
    if band:
        df = df[df['frequency_bins'].between(band[0], band[1])]
    
    # Filter the dataframe for the specified timepoints
    df_tp1 = df[df['time_points'] == tp1].reset_index(drop=True)
    df_tp2 = df[df['time_points'] == tp2].reset_index(drop=True)
    print('Confirm the change is well computed: Original heads vs. change:')
    print(df_tp1.head(), df_tp2.head())
    
    if len(df_tp1) != len(df_tp2):
        raise ValueError(f"Different number of samples for '{tp1}' and '{tp2}'")
    
    # Compute the change in power
    changes = df_tp2['power'].values - df_tp1['power'].values 
    
    print(changes)
          
    # Calculate average and median change
    average_change = np.mean(changes)
    median_change = np.median(changes)
    
    # Save the results
    results = {
        'Changes': changes,
        'Average Change': average_change,
        'Median Change': median_change
    }
    
    # Define the bin range using the provided tuple (start, stop, step)
    bin_edges = np.arange(bins[0], bins[1] + bins[2], bins[2])

    # Plot the distribution of changes using a step-style histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(changes, element='step', bins=bin_edges, color=color, edgecolor=color, alpha=0.3)
    print(bin_edges)
    # Add vertical lines for zero and average change
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(average_change, color=color, linestyle=':', linewidth=2)
    plt.axvline(median_change, color=color, linestyle='-', linewidth=2)
    plt.axvline(x=1.8, color='darkgoldenrod', linestyle='-',  linewidth=1)
    # Add plot labels and title
    band_label = f" ({band[0]}-{band[1]} Hz)" if band else ""
    plt.title(f"Distribution of Power Changes: {tp1} vs {tp2}{band_label}")
    plt.xlabel('Z-scored power change')
    plt.ylabel('Count')
    
    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    # Clean up the plot's aesthetic
    sns.despine()
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Define the directory and filename
    directory = '/Users/bgoncalves/Desktop/Sofia - PhD/SVG Figures'
    filename = 'F5_B.svg'

    # Save the figure as an SVG file
    fig = plt.gcf()  # Get the current figure
    fig.savefig(os.path.join(directory, filename), format='svg')

    # Show the plot
    plt.show()
    
    return results, df_tp1, df_tp2

        
    
def save_figure(directory, filename_base, dpi=300):
    """
    Save the current figure as both SVG and PNG files in the specified directory.
    Arguments:
    directory - The directory where the files will be saved.
    filename_base - The base name for the files (without extension).
    dpi - The resolution of the PNG file (default is 300).
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construct file paths
    svg_path = os.path.join(directory, f"{filename_base}.svg")
    png_path = os.path.join(directory, f"{filename_base}.png")
    # Save as SVG
    plt.savefig(svg_path, format='svg', bbox_inches='tight')    
    # Save as PNG
    plt.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')
    print(f"Figures saved as {svg_path} and {png_path}")
    
    
    
  