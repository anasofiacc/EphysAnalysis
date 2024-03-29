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
        ax[i].plot(chunk_df.loc[xlim0:xlim1, channels[3]], linewidth=.6)
        ax[i].plot(swr_filt_chunk_df.loc[xlim0:xlim1, channels[3]]+400, color='red', linewidth=.6)
        ax[i].plot(unit_filt_chunk_df.loc[xlim0:xlim1, channels[3]]+600, color='green', linewidth=.6)
        ax[i].set_ylabel('TT{}'.format(tts[i]))
        sns.despine()

        