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


def get_data_filenames(path):
    
    file_list = get_file_list(path, "*.dat")
    amplifier_file=[f for f in file_list if 'amplifier' in f][0]
    ttl_input_file=[f for f in file_list if 'ttl' in f][0]
    timestamp_file = [f for f in file_list if 'rhd2000' in f][0]
    
    print('Amplifier file: {}'.format(amplifier_file))
    print('TTL input file: {}'.format(ttl_input_file))
    print('TTL input file: {}'.format(timestamp_file))
    
    return amplifier_file, ttl_input_file, timestamp_file


def calculate_nr_of_blocks(filename, nchannels, chunk_size):

    file_size = os.stat(filename).st_size  # file_size is returned as string
    
    nr_samples = int(file_size) / (nchannels * 2)  # unit16: 1 sample = 2 bytes ( = 16bits)
    print('Number of samples per channel: {}'.format(nr_samples))

    # nblocks results from the division of total nsamples per argument samples_per_block
    nblocks = nr_samples / chunk_size

    # Number of blocks is the product of a division, it needs to be rounded to an integer (the lowest)
    rounded_nblocks = int(nblocks)

    # The number of samples contained in the leftover block is saved in block_leftover
    block_leftover = (nblocks - rounded_nblocks)

    return rounded_nblocks, block_leftover

def open_and_convert_chunk_to_df(fileid, ch_s, nr_channels):
    
    fileid.seek(0, 1)
    # Open chunk
    block = np.fromfile(fileid, np.uint16, ch_s * nr_channels)
    reshaped_block = block.reshape(ch_s, nr_channels)
    
    # Convert chunk into a dataframe(chunk_size * nr_channels)
    df_block = pd.DataFrame(reshaped_block)
    
    #df_block = df_block * 10  # To keep precision of decimal point for dat file
    return df_block


def get_ttl_input_first_index(ttl_input_file):
    
    # Get index of first non-zero element from TTL input file (First image acquired):
    ttlinput_obj = np.fromfile(ttl_input_file, dtype='uint8')
    
    b = np.nonzero(ttlinput_obj)
    first_frame_index = b[0][0]
    
    return first_frame_index

def organize_chunk_by_tt_and_save(path, chunk, count):
    
    # Get tetrode mapping data
    f = glob.glob('*tt_to_channel_mapping.csv')[0]
    tt_mapping = pd.read_csv(f, names=['tt','ch0','ch1','ch2','ch3','bundle'])  

    # Create a folder for each tetrode
    for tt in tt_mapping.tt.unique():
        try:
            tt_path = os.path.join(path, 'TT{}'.format(tt))
            os.makedirs(tt_path)  # Create a folder for each tetrode
        except:
            pass
        
        # Save the TT data inside the folder, into a csv file
        tt_channels=tt_mapping.loc[tt_mapping.tt==tt, ['ch0','ch1','ch2','ch3']].iloc[0].to_list()  
        tt_chunk=chunk.loc[:, tt_channels]  
        filename='TT{}_chunk{}.csv'.format(tt, count)
        tt_chunk.to_csv(os.path.join(tt_path, filename))

def get_file_list(path, filetype):
    
    # Opens a list containing all filenames of file type in path with
    os.chdir(path)

    for filename in sorted(glob.glob(filetype), key=os.path.getmtime):
        # Add all file names from session path to file list

        try:
            file_list.append(filename)

        except:
            file_list = list()
            file_list.append(filename)

    return file_list


def get_chunk_files(path, tts, chunk_nr):
    '''
    Get files in path with chunk nr and tetrode number from tts
    '''
    
    file_ids=['TT{}_chunk{}.csv'.format(tt, chunk_nr) for tt in tts]
    chunk_files=[]
    
    for (path, dirs, files) in os.walk(path):
        for f in files:
            if f in file_ids:
                chunk_files.append(f)
    
    return chunk_files


def read_and_aggregate_chunk(chunk_files, path):
    '''
    Read each tetrode chunk and concatenate all in a dataframe
    '''
    df_list=[]
    for f in chunk_files:
        tt_nr=re.search(r'(TT[0-9]+)', f).group(0)
        
        # Open tetrode chunk
        tt_chunk_df = pd.read_csv(os.path.join(path, tt_nr, f), 
                                  header=None,
                                  names=['ch0', 'ch1','ch2','ch3'],
                                  )
        # Rearrange chunk for plotting
        tt_chunk_stacked =(tt_chunk_df.reset_index(drop=True).stack().reset_index().rename(columns={
            'level_0':'sample_nr', 'level_1':'channel', 0:'value'}))
        
        tt_chunk_stacked.loc[:, 'TT']=tt_nr
        df_list.append(tt_chunk_stacked)

    return pd.concat(df_list)

def band_pass_filter_data(df, lower_lim, upper_lim, order):
    '''
    Band pass the data for given frequencies (Hz) and order values
    df, Pandas DataFrame:  contains the chunk tetrode data
    lower_lim, int:  Lower limit of the frequency to filter (Hz)
    upper_lim, int: Upper limit of the frequency to filter (Hz)
    '''
    df_=df.copy()
    rate = int(30000)
    [b, a] = butter(order, 
                    ((int(lower_lim)/(rate / 2.)), 
                    (int(upper_lim) / (rate / 2.))), 
                    'pass')
    
    filt = filtfilt(b, a, df_, axis=0)
    return filt

def plot_raw_and_swr_filtered_lfp(df_raw, df_swr_filtered, df_spike_filtered, xlim0, xlim1, tt, chunk_nr):
    
    '''
    '''

    channels = df_raw.columns
    sns.set(context='talk', style='white')
    fig, ax = plt.subplots(4,1, squeeze=True, figsize=(18,18), sharey=True)
    fig.suptitle('HIPP tetrode: {}, chunk {}'.format(tt, chunk_nr))
    for i in [0,1,2,3]:
        ax[i].plot(df_raw.loc[xlim0:xlim1,channels[i]], linewidth=.8)
        ax[i].plot(df_swr_filtered.loc[xlim0:xlim1, channels[i]]+600, color='red', linewidth=.8)
        ax[i].plot(df_spike_filtered.loc[xlim0:xlim1, channels[i]]+900, color='green', linewidth=.8)
    
    sns.despine()
    
def plot_chunk_on_multiple_tts(path, tts, chunk, xlim0, xlim1):
    '''
    Plot the data chunk from multiple tetrodes in tts
        path, str: path were the data is stored
        tts, list: contains the tetrode numbers to plot
        chunk, int: number of the chunk to plot
        xlim0, int: start of x limit
        xlim1, int: end of x limit  
    '''
    num_tts=len(tts)

    sns.set(context='talk', style='white')
    fig, ax = plt.subplots(num_tts,1, squeeze=True, figsize=(18,18), sharex=True)

    chunk_files = get_chunk_files(path, tts, chunk)  
    
    for i,f in enumerate(chunk_files):

        chunk_path = os.path.join(path, 'TT{}'.format(tts[i]), f)
        chunk_df = pd.read_csv(chunk_path, index_col=0)

        swr_filt_chunk = band_pass_filter_data(chunk_df, 150, 250, 2)        
        swr_filt_chunk_df = pd.DataFrame(swr_filt_chunk, columns=chunk_df.columns)

        unit_filt_chunk= band_pass_filter_data(chunk_df, 500, 6000, 2)   
        unit_filt_chunk_df=pd.DataFrame(unit_filt_chunk, columns=chunk_df.columns)  

        # Plot 1st channel of each tetrode
        channels=chunk_df.columns
        ax[i].plot(chunk_df.loc[xlim0:xlim1, channels[1]], linewidth=.6)
        ax[i].plot(swr_filt_chunk_df.loc[xlim0:xlim1, channels[1]]+400, color='red', linewidth=.6)
        ax[i].plot(unit_filt_chunk_df.loc[xlim0:xlim1, channels[1]]+600, color='green', linewidth=.6)
        ax[i].set_ylabel('TT{}'.format(tts[i]))
        sns.despine()
    
#------------------------ PRE-PROCESSING FOR SHARP WAVE RIPPLES DETECTION ----------------------------------#


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
        
        files = get_file_list(tt_path, "*_swr_filtered.csv")  
        #Append file paths to list
        for f in files:
            filtered_files_paths.append(os.path.join(tt_path, f))
    return filtered_files_paths


def open_swr_filtered_data(filtered_files_paths):
    
    '''
    Opens the files in filtered_files_paths and stores them in a dictionary.
        filtered_files_paths, list: contains paths to swr filtered files to open
    Returns:
        dfs, dict: chunk numbers as keys. Each chunk key contains a dictionary with tts as keys
    '''
    
    dfs = {}
    
    # Open files in filered_files list.
    for p in tqdm(filtered_files_paths):
        print(p)
        #Get chunk number and tetrode label from file path
        chunk_nr = re.search(r'chunk([0-9]+)', p).group(1)
        tt = re.search(r'TT([0-9]+)_', p).group(1)

        # Open the data file
        data = pd.read_csv(p)

        #Create keys if not existent
        if chunk_nr not in dfs.keys():
            dfs[chunk_nr]={}
        if tt not in dfs[chunk_nr].keys():
            dfs[chunk_nr][tt]=0

        # Append to a dictionary containing chunk nr and tt as keys
        dfs[chunk_nr][tt]=data

    return dfs


def pre_process_data_for_swr_detection(dfs):
    '''
    Pre-process the each data chunk for SWR detection (as described by Kai et al.(2016))
        dfs, dict: dictionary of dictionaries split by chunks and pyr tetrodes.
    Returns:
        summed_dfs, dict: dictionary storing chunks of squared summed data across tetrodes
        smoothed_dfs, dict: similar, with smoothed data
        sqrt_dfs, dict: similar, with square-root after smoothing
    '''
    
    # For each chunk, sum the dataframes
    squared_summed_chs_dfs = {}
    summed_dfs={}
    smoothed_dfs={}
    sqrt_dfs = {}

    for chunk in tqdm(dfs.keys()):
        
        squared_summed_chs_dfs[chunk]={}
        summed_dfs[chunk]=0
        smoothed_dfs[chunk]=0
        sqrt_dfs[chunk]=0

        for tt in dfs[chunk].keys():     
            # Square data and sum channels in each tetrode
            squared_summed_chs_dfs[chunk][tt]=0
            squared_summed_chs_dfs[chunk][tt]=((dfs[chunk][tt])**2).sum(axis=1)   

        # Sum data across tetrodes
        summed_dfs[chunk]=sum(squared_summed_chs_dfs[chunk].values())

        # Smooth the data using a gaussian filter (sigma= 4ms, correspnding to 120 samples)
        smoothed_dfs[chunk]=gaussian_filter1d(summed_dfs[chunk], sigma=120)

        # Square root the data
        sqrt_dfs[chunk]=np.sqrt(smoothed_dfs[chunk]) 

    return summed_dfs, smoothed_dfs, sqrt_dfs

    
def save_pre_processed_data_for_swr_detection(main_path, sqrt_dfs):
    '''
    Save the data pre-processed for SWR detection in a new folder in the main_path
        main_path, str: the path where the all the data from one session is stored
        sqrt_dfs, dict: dictionary containing the data to be saved      
    '''
    
    #Create a folder to contain the processed data
    new_folder_path = os.path.join(main_path, 'Preprocessed for SWR detection')
    try:
        os.mkdir(new_folder_path)
    except:
        pass
    
    # Store the data into chunks
    for chunk in tqdm(sqrt_dfs.keys()):
        filename='chunk{}_preprocessed_swr.csv'.format(chunk) 
        pd.DataFrame(sqrt_dfs[chunk]).to_csv(os.path.join(new_folder_path, filename),
                           index=False)

def plot_processing_steps(summed_dfs, smoothed_dfs, sqrt_dfs, chunk_nr, start, end):
    '''
    Plots the data from chunk (chunk nr) between start and end, each processing step
        summed_dfs, dict: dictionary storing chunks of squared summed data across tetrodes
        smoothed_dfs, dict: similar, with smoothed data
        sqrt_dfs, dict: similar, with square-root after smoothing
        chunk_nr, str: number of the chunk to plot
        start, int: start of the range to plot on the x axis
        end, int: end of the range to plot on the x axis
    '''
    
    # Plot parts of two chunks
    fig, ax = plt.subplots(3,1, figsize=(25,10))
    
    # Plot parts of two chunks
    ax[0].plot(summed_dfs[chunk_nr][start:end])
    ax[1].plot(smoothed_dfs[chunk_nr][start:end])
    ax[2].plot(sqrt_dfs[chunk_nr][start:end])
    
    sns.despine()
    
def calculate_average_and_std_of_pre_processed_data(sqrt_dfs):
    
    '''
    Calculate the average and standard deviation of all data
        sqrd_dfs, dict: contains the pre-processed data
    Returns:
        signal_average, float: the average of the pre-processed data
        singal_std, float: the standard deviation of the pre-processed data
    
    '''
    
    means=[]
    stds = []
    for c in sqrt_dfs.keys():
        means.append(np.mean(sqrt_dfs[c]))
        stds.append(np.std(sqrt_dfs[c]))

    # Calculate the signal average
    signal_average=np.mean(means)
    # Calculate the signal standard deviation
    signal_std = np.mean(stds)
    
    return signal_average, signal_std



def detect_threshold_crossings(sqrt_dfs, threshold):
    '''
    Detect threshold crossings in all chunks and store them into a dataframe
    sqrt_dfs, dict: dictionary with pre-processed data chunks (keys)
    threshold, int: threshold
    Return:
        all_crossings, pandas DataFrame - threshold crossings
    '''
    all_crossings=[]

    for chunk in sqrt_dfs.keys():

        # Convert to a pandas dataframe
        sqrt_dfs[chunk] = pd.DataFrame(sqrt_dfs[chunk]).rename(columns={0:'voltage'})

        # Create new column that tags voltage values above the threshold (boolean mask):
        sqrt_dfs[chunk]['above_threshold'] = (sqrt_dfs[chunk].voltage > threshold)

        # Find crossings
        sqrt_dfs[chunk]['cross'] = (sqrt_dfs[chunk].above_threshold.diff().fillna(0)!=0)
        crosses=sqrt_dfs[chunk][sqrt_dfs[chunk]['cross']==1].reset_index()

        #Add crossings to dataframe
        chunk_crossings=pd.DataFrame()
        chunk_crossings['crosses']=crosses.loc[:,'index']
        chunk_crossings['chunk']=int(chunk)
        all_crossings.append(chunk_crossings)

    return (pd.concat(all_crossings)).reset_index(drop=True)



def plot_chunk_on_pyr_tts_and_swr_detection(path, 
                                            tts, 
                                            chunk, 
                                            sqrt_dfs, 
                                            swrs, 
                                            t, 
                                            xlim0, 
                                            xlim1):
    '''
    Plot the data chunk from pyramidale tetrodes in tts
        path, str: path were the data is stored
        tts, list: contains the tetrode numbers to plot
        chunk, int: number of the chunk to plot
        sqrt_dfs, dict: chunks as keys. Contains processed data for SWR detection
        swrs, pandas dataframe: contains the putative SWRs
        t, int: threshold of SWR detection
        xlim0, int: start of x limit
        xlim1, int: end of x limit  
    '''
    num_tts=len(tts)

    sns.set(context='talk', style='white')
    fig, ax = plt.subplots(num_tts+1,1, squeeze=True, figsize=(18,18), sharex=True)

    chunk_files = get_chunk_files(path, tts, chunk)  
    
    for i,f in enumerate(chunk_files):

        chunk_path = os.path.join(path, 'TT{}'.format(tts[i]), f)
        chunk_df = pd.read_csv(chunk_path, index_col=0)

        swr_filt_chunk = band_pass_filter_data(chunk_df, 150, 250, 2)        
        swr_filt_chunk_df = pd.DataFrame(swr_filt_chunk, columns=chunk_df.columns)

        unit_filt_chunk= band_pass_filter_data(chunk_df, 500, 6000, 2)   
        unit_filt_chunk_df=pd.DataFrame(unit_filt_chunk, columns=chunk_df.columns)  

        # Plot 1st channel of each tetrode
        channels=chunk_df.columns
        ax[i].plot(chunk_df.loc[:, channels[1]], linewidth=.6)
        ax[i].plot(swr_filt_chunk_df.loc[:, channels[1]]+400, color='red', linewidth=.6)
        ax[i].plot(unit_filt_chunk_df.loc[:, channels[1]]+600, color='green', linewidth=.6)
        ax[i].set_ylabel('TT{}'.format(tts[i]))
        
        for c in swrs.loc[swrs['chunk']==chunk, 'crosses']:
            ax[i].axvline(c, color='orange')  
        
    ax[i+1].plot(sqrt_dfs[str(chunk)]['voltage'])
    ax[i+1].axhline(t, color='green')
    plt.xlim([xlim0, xlim1])
    sns.despine()