import pandas as pd
import os
import numpy as np
import seaborn as sns
import glob
import re
from tqdm import tqdm
from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ephys_utils import *
 
        
#------------------------ PRE-PROCESSING FOR SHARP WAVE RIPPLES DETECTION ----------------------------------#

def calculate_velocity(path):

    # Open the position data file
    position_filename = glob.glob(os.path.join(path, 'Timestamped_position','*_clean.csv'))
    position = pd.read_csv(position_filename[0])

    # Calculate euclidean distance 
    position['y_diff']=(position['y'].diff()).fillna(0)
    position['delta_t']=(position['timestamp'].diff()).fillna(0)
    position['delta_d'] = np.sqrt(position['x_diff']**2 + position['y_diff']**2)
    position['vel']=position.delta_d / position.delta_t
    position.loc[abs(position.vel)>100, 'vel']=np.nan
    print('\n Number of points to interpolate: {}'.format(position.vel.isna().sum()))

    # Interpolate NaNs using linear interpolation (limit=1)
    position.vel.interpolate(limit=5, inplace=True)

    # Save into a .csv file
    position.to_csv(position_filename[0], index=False)

    print('\n Non interpolated points: {}'.format(position.vel.isna().sum()))
    position.vel.hist(bins=50)
    
    return position


def concatenate_velocity_and_ephys_timestamps(path, position):

        timestamp_path = os.path.join(path, 'Ephys_timestamps')
        timestamp_files = get_file_list(timestamp_path, "*.csv")

        all_timestamp_chunks = []
        for t in timestamp_files:

            # Open timestamp file
            tfile_path = os.path.join(timestamp_path, t)
            t_chunk = pd.read_csv(tfile_path, index_col=0).rename(columns={'0':'timestamp'}).reset_index(
                drop=True)

            all_timestamp_chunks.append(t_chunk)

        # Concatenate the ephys timestamps into a single dataframe
        ephys_timestamps=pd.concat(all_timestamp_chunks, axis=0)

        # Add velocity to timestamps
        timestamps_merged=pd.concat([ephys_timestamps, position[['timestamp', 'vel', 'x', 'y']]])
        timestamps_sorted=timestamps_merged.sort_values(by=['timestamp']).reset_index(drop=True)
        
        # Remove duplicated timestamps with NaNs
        duplicated_timestamps=timestamps_sorted.loc[timestamps_sorted.duplicated(), 'timestamp'].unique()
        
        # This will make the duplicated timestamp also Nan so we can remove it using dropna()
        timestamped_xy_vel = timestamps_sorted.groupby(['timestamp']).first().reset_index()
        
        return timestamped_xy_vel, ephys_timestamps

def interpolate_velocity(timestamped_xy_vel, xlim1, xlim2):
        
    # Interpolate missing values
    timestamped_xy_vel[['vel_interp', 'x_interp', 'y_interp']] = timestamped_xy_vel[
    ['vel', 'x', 'y']].interpolate(method='linear')
    
    plt.figure(figsize=(20,5))  
    plt.scatter(timestamped_xy_vel['timestamp'], timestamped_xy_vel['vel_interp'], color='blue', s=.2) 
    plt.scatter(timestamped_xy_vel['timestamp'], timestamped_xy_vel['vel'], color='red', s=10) 
    plt.xlim([150,160])
    plt.ylim([0,80])
    sns.despine()
    
    return timestamped_xy_vel
    
  
def band_pass_filter_data(df, lower_lim, upper_lim, order, rate):
    '''
    Band pass the data for given frequencies (Hz) and order values
    df, Pandas DataFrame:  contains the chunk tetrode data
    lower_lim, int:  Lower limit of the frequency to filter (Hz)
    upper_lim, int: Upper limit of the frequency to filter (Hz)
    '''
    df_=df.copy()
    [b, a] = butter(order, 
                    ((int(lower_lim)/(rate / 2.)), 
                    (int(upper_lim) / (rate / 2.))), 
                    'pass')
    
    filt = filtfilt(b, a, df_, axis=0)
    return filt


def band_pass_filter_ephys(path, tts, rate, ephys_timestamps,timestamped_xy_vel):
    '''
    Opens each pre-processed ephys chunk, for each terode. Concatenates the chunks.
    Band-pass filters the data, adds the timestamps and saves into a .csv file
    '''
    
    for tt in tqdm(tts): 
        print(tt)
        # Get file list
        tt_path=os.path.join(path, 'TT{}'.format(tt))
        ephys_files = get_file_list(tt_path, "*.csv")   
        
        # Remove files with "immobility" in their name. 
        # Otherwise they will be processed
        to_process = [f for f in ephys_files if "timestamped" not in f]
          
        tt_ephys_data=[]   
        for e in to_process:
            
            # Read each ephys raw data file
            efile_path = os.path.join(tt_path, e)
            # Get chunk number of use to know from which file the putative events came
            chunk = re.search(r'chunk[0-9]*', efile_path)[0]
            ephys_chunk = pd.read_csv(efile_path, index_col=0)
            ephys_chunk = ephys_chunk.reset_index()
            ephys_chunk['chunk'] = chunk
            ephys_chunk.rename(columns={'index':'chunk_index'}, inplace=True)
            
            # Append data from each file to list
            tt_ephys_data.append(ephys_chunk)
        
        # Concatenate list into a dataframe
        tt_ephys=(pd.concat(tt_ephys_data)).reset_index(drop=True)
        print('-- Filtering --')
        to_filter = tt_ephys.drop(['chunk', 'chunk_index' ], axis=1)
        filtered = band_pass_filter_data(to_filter, 150, 250, 2, rate)        
        filtered_df = pd.DataFrame(filtered, columns=to_filter.columns)
        
        # Re-add chunk data 
        filtered_df['chunk']=tt_ephys['chunk']
        #Re-add index 
        filtered_df['chunk_index']=tt_ephys['chunk_index']
        
        # Add timestamps
        tt_ephys_concat = pd.concat([filtered_df.reset_index(drop=True), 
                                     ephys_timestamps.reset_index(drop=True)], axis=1) 
        
        # Add velocity data (merging through timestamps)
        tt_ephys_combined = tt_ephys_concat.merge(timestamped_xy_vel, 
                                                  how='left', on='timestamp') 
        #Save immobility filtered ephys data
        print('-- Saving --')
        filename = 'TT{}_timestamped_filtered.csv'.format(tt)
        tt_ephys_combined.to_csv(os.path.join(tt_path, filename), index=False)
        
        #return tt_ephys_combined
 

def get_swr_filtered_files_for_pyr_tetrodes(main_path, pyr_tts):
    
    '''
    Get the names of the SWR filtered files in the tetrode folders in pyr_tt
        main_path, str: the path where the all the data from one session is stored
        pyr_tts, list: contains the numbers of the pyramidale tetrodes
    Returns:
        filtered_files_paths, list: contains the paths to each filtered file in the indicated TT folders
    '''
    # Collect the filtered file names in pyr tts folders
    filtered_files_paths = []
    for tt in pyr_tts:
        # Create tt_path and append to list
        tt_path = os.path.join(main_path,'TT{}'.format(tt)) 
        # Get filtered data file names and store them in a list
        
        files = get_file_list(tt_path, "*timestamped_filtered.csv")
        
        #Append file paths to list
        for f in files:
            filtered_files_paths.append(os.path.join(tt_path, f))
    return filtered_files_paths


def open_swr_filtered_data(filtered_files_paths):
    
    '''
    Opens the files in filtered_files_paths and stores them in a dictionary.
        filtered_files_paths, list: contains paths to swr filtered files to open
    Returns:
        dfs, dict: with tts as keys
    '''
    
    dfs = {}
    
    # Open files in filered_files list.
    for p in tqdm(filtered_files_paths):
    
        # Get tetrode label from file path
        tt =int(re.search(r'TT([0-9]+)_', p).group(1))
        
        # Open the data file
        data = pd.read_csv(p)

        # Append to a dictionary containing tt as keys
        dfs[tt]=data

    return dfs

def pre_process_data_for_swr_detection(dfs):
    '''
    Pre-process the each tetrode data for SWR detection (as described by Kai et al.(2016))
        dfs, dict: dictionary of dictionaries split by pyr tetrodes.
    Returns:
        summed_data, Series: Data summed across tts
        smoothed_data, Series: Data smoothed
        sqrt_data, Series: Data square-rooted       
    '''
    
    tt_df_list=[]
    
    for tt in tqdm(dfs.keys()):
        cols_to_process=[col for col in dfs[tt].columns if col not in 
                     ['chunk_index', 'chunk', 'index', 'timestamp', 'vel', 'x', 'y', 'vel_interp','x_interp', 'y_interp']]
        
        # Square the data and sum tt channels
        tt_df_list.append(((dfs[tt][cols_to_process])**2).sum(axis=1)) 
        
    # Sum data across tetrodes
    summed_data=sum(tt_df_list)
    
    # Smooth the data using a gaussian filter (sigma= 4ms, correspnding to 12 samples)
    smoothed_data= summed_data.rolling(window=60, win_type='gaussian', center=True).mean(std=12)
    
    # Square root the data
    sqrt_data=np.sqrt(smoothed_data) 
    
    return summed_data, smoothed_data, sqrt_data

    
def plot_processing_steps(summed_data, smoothed_data, sqrt_data, start, end):
    '''
    Plots the data between start and end, each processing step
        summed_data, Series: squared summed data across tetrodes
        smoothed_data, Series: similar, with smoothed data
        sqrt_data, Series: similar, with square-root after smoothing
        start, int: start of the range to plot on the x axis
        end, int: end of the range to plot on the x axis
    '''
    
    # Plot parts of two chunks
    fig, ax = plt.subplots(3,1, figsize=(25,14))
    
    # Plot parts of two chunks
    ax[0].plot(summed_data[start:end])
    ax[1].plot(smoothed_data[start:end])
    ax[2].plot(sqrt_data[start:end])
    sns.despine()
            
        
def calculate_threshold(df):
    
    '''
    Calculate the threshold using the average and standard deviation
        df, dataframe: containing the ephys data
    Returns:
        threshold, float: the detection threshold
        avg, float: signal average of the pre-processed data    
    '''
    
    # Calculate the signal average
    avg =np.mean(df['ephys'])
    # Calculate the signal standard deviation
    std_ = np.std(df['ephys'])
    threshold= avg + 2*std_
    
    return threshold, avg



def detect_threshold_crossings(df, threshold):
    '''
    Detect threshold crossings in df
    df DataFrame: processed data
    threshold, int: threshold
    Return:
        crossings, pandas DataFrame - threshold crossings
    '''
    df_=df.copy()
    
    # Create a boolean mask of square-rooted data above threshold
    df_['above_thresh_mask'] = (df_['ephys'] > threshold)
    
    # Find crossings (where difference to previous != 0)
    df_['crosses'] = df_['above_thresh_mask'].diff().fillna(0)!=0
    
    # Get ephys values prior to crossing
    df_['prev_ephys']= df_['ephys'].shift()

    # Keep crossings
    crossings = df_[df_['crosses']==True]
    
    # classify crossings according to upward or downward movement
    crossings['cross_type']=np.where(crossings.ephys < crossings.prev_ephys, 'descending', 'ascending')

    return crossings

def get_respective_descending_crossing(cross, df, threshold):
    '''
    For each ascending threshold crossing found in immobility, get the respective
    descending crossing (searching on the time window of 200 ms that follow the ascending
    crossing). 
    '''
    
    # Get timestamp
    t = cross.timestamp
    # Get subsection
    subsection = df[df.timestamp.between(t, t+.2)]
    # Detect crossing in subsection
    crossing = detect_threshold_crossings(subsection, threshold)
    return crossing.timestamp.iloc[0]

def get_max_voltage_for_each_event(g, df):
    '''
    Get the maximum voltage in each putative event.
    g, Pandas Series: contains event data
    df, dict: contains voltage data
    Returns: max_v, float: maximum voltage in event
    '''
    g_=g.copy()
    
    # Get maximum voltage between cross start and end
    max_v = df.loc[df.timestamp.between(
        g_.loc['timestamp'],
        g_.loc['threshold_descent_timestamp']), 'ephys'].max()
    
    return max_v



def get_swr_onset_and_offset(event, filtered, data_avg, cross_col):
    '''
    Get onset and offset of detected events using data average as threshold crossing
        event, series: the event 
        filtered, pandas: processed ephys data
        data_avg, int: the data average
        cross_col, str: which index name we will be used as cross reference:
        index for onset and shifted_index for offset
        
    Returns: onset or offset for each cross    
    '''   
    
    # Get timestamp of crossing
    t = event[cross_col]
    
    # Get subsection of data to search for voltage values above average
    if cross_col=='timestamp': 
        # Search within previous .5 seconds
        subsection = filtered.loc[filtered.timestamp.between(t-.2,t)]

    elif cross_col=='threshold_descent_timestamp':
         subsection = filtered.loc[filtered.timestamp.between(t,t+.2)]    
        
    # Create new column that tags voltage values above the threshold (boolean mask):
    subsection['above_avg'] = (subsection.ephys > data_avg)
    
    # Find crossings 
    subsection['avg_cross'] = (subsection.above_avg.diff().fillna(0)!=0)
    crosses=subsection[subsection['avg_cross']==1].reset_index()
    
    # Get closest above average crossing
    closest_indices = crosses['timestamp'].sub(event[cross_col]).abs().sort_values().index.to_list()
   
    # If no closest crossing if found
    if not closest_indices:
        return np.nan
    else:
        closest_index=closest_indices[0]
        return crosses.loc[closest_index, 'timestamp'] 
    
    
def add_timestamps(group, timestamps_path):
    
        '''
        Add the timestamps to the group's SWRs
        '''
        g_ = group.copy()
        
        # Open timestamps of chunk
        chunk_nr = g_.chunk.iloc[0]
        f = os.path.join(timestamps_path,'timestamps_chunk{}.csv'.format(chunk_nr))
        t = pd.read_csv(f)
        g_['onset_timestamp']=g_['onset'].apply(lambda x: t.iloc[int(x)][0])
        g_['offset_timestamp']=g_['offset'].apply(lambda x: t.loc[int(x)][0])
        
        return g_
    
    
    
def sanity_check_for_swr_detection(path, dfs, events, data, tt, event_nr, ch, t_range):
    '''
    dfs, dataframe - contains the raw data
    data, dataframe - contains pre-processed data 
    events, dataframe
    tt, int - the tetrode used to plot
    event_nr, int - the number of the detected putative swr event   
    ch, int: ranges from 1 to 4. Channel of the tetrode
    t_range, int - to create the plot boundaries. In time (secs)
    '''
    # CREATE FIG
    fig, axs = plt.subplots(3,1, dpi=200, figsize=(14,8), 
                            gridspec_kw={'height_ratios': [.5, 1, 2]}, sharex=True)
    
    # GET INFO
    chunk = events['chunk'].iloc[event_nr]
    ascend_cross_t = round(events['ascend_cross_t'].iloc[event_nr], 6)
    onset_t = round(events['onset_t'].iloc[event_nr], 6)
    offset_t = round(events['offset_t'].iloc[event_nr], 6)
    print('timestamp: {}, onset: {}'.format(ascend_cross_t, onset_t))
    
    # VELOCITY DATA AND FILTERED EPHYS --------------------------------------------------------
    filtered_immob = dfs[tt].loc[dfs[tt]['timestamp'].between(
        ascend_cross_t-t_range, ascend_cross_t+t_range
    ), :]
    
    # Plot velocity data
    axs[0].plot(filtered_immob['timestamp'], filtered_immob['vel_interp'], color='black', linewidth=1)
    axs[0].axhline(4, linestyle='dashed', color='red', linewidth=1)
    axs[0].set_ylabel('velocity')
    
    #Plot filtered ephys data
    axs[2].plot(filtered_immob['timestamp'], filtered_immob.iloc[:, ch]+400, 
                color='red', linewidth=2) 
    
    # EPHYS TIMESTAMPS ------------------------------------------------------------------
    # Open timestamps data
    tchunk_path = os.path.join(path, 'Ephys_timestamps', 'timestamps_{}.csv'.format(chunk))
    timestamps = pd.read_csv(tchunk_path)
    
    # Get timestamps of interest (toi)
    ephys_t_mask = timestamps['0'].between(ascend_cross_t-t_range, ascend_cross_t+t_range)
    toi = timestamps.loc[ephys_t_mask,'0']

    # RAW EPHYS ------------------------------------------------------------------------
    
    # Get raw data for 1 TT channel tt in chunk   
    echunk_path = os.path.join(path, 'TT{}'.format(tt), 'TT{}_{}.csv'.format(tt, chunk))
    channel_data = pd.read_csv(echunk_path).iloc[:,ch+1]
    
    # Plot raw ephys data
    axs[2].plot(toi, channel_data[ephys_t_mask], linewidth=.8, color='black')
    axs[2].axvline(ascend_cross_t, linestyle='dotted', color='green', linewidth=2)
    axs[2].set_ylim([-500, 600])
    
    # PRE-PROCESSED DATA ---------------------------------------------------------------- 
    section = data.loc[data.timestamp.between(ascend_cross_t-t_range, ascend_cross_t+t_range)]
    axs[1].plot(section['timestamp'], section['ephys'])
    # Add threshold
    axs[1].axhline(threshold, color='green')
    #Add onset
    axs[1].axvline(onset_t, linestyle='dotted', color='red')
    # Add offset
    axs[1].axvline(offset_t, linestyle='dotted', color='blue')
    # Add signal average
    axs[1].axhline(data_avg, color='red')
    # Add threshold
    axs[1].axvline(ascend_cross_t, linestyle='dotted', color='green', linewidth=2, alpha=1)
    axs[1].set_ylim([0,300])
    sns.despine()