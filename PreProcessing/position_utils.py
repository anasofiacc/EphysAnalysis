import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector
import seaborn as sns


def collect_and_organize_position_data(files_path, diff, interpolated=5):

    '''

    :param files_path: directory of the files containing the timestamps and position to be analyzed.
    :param diff: The x position difference to be accounted in order to divide runs;
    : param interpolated: The number of interpolated points

    :return:  MultiIndex DataFrame with the levels session, run number and index. Contains x and y position and
            respective timestamps. Also included information about
            1)run type; 2) stimulation condition; 3) outcome information if attribute_specs_to_runs == True
    '''

    if isinstance(files_path, str):
        # Get all .csv files in directory (= files_path). Returns as Series.
        csv_files = get_file_list(files_path, "*.csv")
        
        # Collect .csv files of interest (timestamp, x and y)
        timestamp_file = [f for f in csv_files if 'tstamp_image.' in f][0]
        xcoord = [f for f in csv_files if 'xcoord' in f][0]
        ycoord = [f for f in csv_files if 'ycoord' in f][0]
        
        timestamped_position, session_code, rat_code = collect_data_from_file_into_df(
            files_path, timestamp_file, xcoord, ycoord, diff, interpolated)
        timestamped_position['session'] = session_code
        timestamped_position['rat'] = rat_code
          
    return timestamped_position


def collect_data_from_file_into_df(files_path, timestamp_filename, xcoord_filename, 
                                   ycoord_filename, diff, interpolated):
    '''
    Opens xy position and respective timestamps files. Stores data into a dataframe.
    Data is classified according to each rat run. Position and timestamp data is numbered according to run number.
    :param files_path: directory path to files to be open
    :param timestamp_filename: file name containing xy position timestamps of given session
    :param xcoord_filename: file name containing x position data of given session
    :param ycoord_filename: file name containing y position data of given session

    :return: Dataframe with data from one session, divided into 5 columns: timestamp, x, y, x_diff, run_nr
    '''

    # Find session code in position filename.
    match = re.search(r"(\d)+", timestamp_filename)
    session_code = match.group(0)

    # Find rat code in files_path.
    match = re.search(r"([A-Z]+)_", files_path)
    rat = match.group(1)
    print('Session code: {}, Rat code: {}'.format(session_code, rat))

    # Reads the data in the csv files into a dataframe
    timestamps = pd.read_csv(os.path.join(files_path, timestamp_filename),
                             header=None, names=['timestamp'], delim_whitespace=True)

    xcoord = pd.read_csv(os.path.join(files_path, xcoord_filename),
                           header=None, names=['x'], delim_whitespace=True)
    ycoord = pd.read_csv(os.path.join(files_path, ycoord_filename),
                           header=None, names=['y'], delim_whitespace=True)
    

    print('\n Opening timestamps:%s. Length:%d'%(timestamp_filename, len(timestamps)))
    print('\n Opening  x position:%s. Length:%d'%(xcoord_filename, len(xcoord)))
    print('\n Opening y position:%s. Length:%d\n'%(ycoord_filename, len(ycoord)))
    
    # Concatenate the position data
    position = pd.concat([xcoord, ycoord], axis=1)
    
    # Wrangles position and timestamp data to return a concatenated single dataframe
    timestamped_position = organize_session_data(timestamps, position, session_code, rat, diff, interpolated)

    return timestamped_position, session_code, rat


def organize_session_data(timestamps, position, session_code, rat, diff, interpolated):

    '''
    Performs several data wrangling processes:
        - Interpolation of NaNs and zeros in position data
        - Check for length discrepancies in same session position and timestamp data.
        - Adds column ['run_nr'] that classifies data according to run number

    :param timestamps: session timestamp xy data
    :param position: session xy position data
    :param session_code: session code that identifies the session data.

    :return: Clean, interpolated single dataframe containing timestamps and xy position and datapoints labelled
    according to run number.
    '''

    # Interpolate NaN and zeros (loss of position data) in position dataframe
    position = (position.replace(0, np.nan)
                        .interpolate(method='linear', limit=interpolated)
                        .fillna(0))

    # Number of data points in each session, in the position frame and timestamp frame must be equal
    check_assertion_error(len(timestamps) == len(position),
                          '\n N points in session %s is different!\n')

    # Convert timestamps to ms and reference it to the first timestamp (1st timestamp = 0)
    first_timestamp = timestamps.iloc[0]
    timestamps_conv = (timestamps - first_timestamp) * 1e-7

    # Concatenate timestamps and position into a single 3 column, 2 x levels index DataFrame.
    timestamped_position = pd.concat([timestamps_conv, position], axis=1)

    # Converts xy from pixels to cm. Then Label data points according to run number.
    for col in ['x', 'y']:
        timestamped_position[col] = timestamped_position[col].astype(float)
        timestamped_position[col] = (timestamped_position[col]*10)/50
    
    timestamped_position = timestamped_position[timestamped_position['x']<300]
    
    #Divide into runs
    timestamped_position_div=divide_into_runs(timestamped_position, diff)
    
    return timestamped_position_div


def divide_into_runs(df,diff): 
    '''
    Labels each data point (timestamp - x - y) by run number. Each run the rat performs - sample or test - will be
    detected by consecutive x differences. Differences above a certain distance (in cm) will be considered a
    transition between runs and labelled accordingly with number starting in 1.
    :param df: dataframe with xy timestamped position data.
    :param diff: int, run break difference
    :return: dataframe with xy timestamped position data labelled according to run number
    '''

    # Calculate difference between consecutive elements in 'x' and store them in new column 'x_diff'
    df['x_diff'] = (df['x'].diff()).fillna(0)
    
    # Get the run breaks, including the first and last index of the session
    run_breaks = df.head(1).index.tolist()+(df[df['x_diff']< float(diff)]).index.tolist()+df.tail(1).index.tolist()

    run_number = 1
    df['run_nr'] = np.NaN
    for i, j in zip(run_breaks, run_breaks[1:]):

        df.loc[i:j, 'run_nr'] = run_number
        run_number += 1

    return df


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


def save_individual_runs_data(df, files_path):
    # Create a folder named "Timestamped_Position" inside the files path
    try:
        os.mkdir(os.path.join(files_path, "Timestamped_position"))
    except:
        pass
    
    session = df.session.loc[0]

    path = os.path.join(files_path,"timestamped_position", "%s_timestamped_position_df_raw.csv"%session)
    df.to_csv(path, header=True)
    
def visual_check_of_individual_runs(files_path, diff, interpolated=5):
    
    # Collect a dataframe with position data from all sessions.
    data = collect_and_organize_position_data(files_path, diff, interpolated)
    # Drop timestamps with NaN
    data = data.dropna(subset=['timestamp'])
    
    print(data.head())
    
    # Save the data
    save_individual_runs_data(data, files_path)
    
    # Plot the individual runs
    sns.set(style='white', context='talk')
    g = sns.relplot(kind='scatter', data=data, x='x', y='y', col='run_nr', col_wrap=4)
    plt.show()   
    

def check_assertion_error(statement, printed_output):

    try:
        assert statement

    except AssertionError:

        print(printed_output)

def select_and_save_frame_from_videos_in_path(path):
    
    '''
    Find first frame of each video a save it as a .jpg image
    '''
    # Get the video list
    video_list = get_file_list(path, "*.avi")
    
    # Create each video path
    for video in video_list:
        
        video_path = os.path.join(path, video)
      
        # Open the video
        cap = cv2.VideoCapture(video_path)
            
        # Read the first frame of the video (usually the brighter)
        cap.set(1, 0)
        success, image = cap.read()
            
        # Saves the frame with video tag and frame number
        match = re.search(r"(.+)(?=.avi)", video)
            
        frame_path = os.path.join(
                 path, 
                "ROI_Frames", 
                "%s_1stframe.jpg"%match.group(0)
        )
                
        cv2.imwrite(frame_path, image)
        cap.release() 
        
def collect_rois_df(video_files_path, filename):

    '''
    Opens all .jpg frames from the given path and allows the selection of ROIs upon mouse click.
    Click enter to lock the ROI selection and 's' to stop the loop.
    '''
    
    frame_path = os.path.join(video_files_path, "ROI_Frames")
    frame_list = get_file_list(frame_path, "*.jpg")
    
    for frame in frame_list:
        
        # Open image object (an numpy array)
        single_frame_path = os.path.join(frame_path, frame)
        img = cv2.imread(single_frame_path)
        print(single_frame_path)
        # Select a ROI using the mouse and save coordinates in tuple and add it to a list
        roi = cv2.selectROI(str(frame), img)
        print(roi)
        session=re.search(r"(\d+)",frame)
        roi=roi+(session.group(0),) 
            
        # Keep the image window open until a key (any key is pressed). If 's' key is pressed, 
        # break the loop and empty all_rois 
        k = cv2.waitKey(0)  
        if k:
            cv2.destroyWindow(str(frame))     
            if k==115:
                break      
        
    cv2.destroyAllWindows()
    
    #Convert into a DataFrame and save it into a .csv file
    roi_df = pd.DataFrame([roi],columns=['x', 'y', 'width', 'height', 'session'])                    
    
    path=os.path.join(video_files_path, "ROIS", "%s_%s.csv"%(session.group(0), filename))
    roi_df.to_csv(path,header=True)
    
    return roi_df, session.group(0)

def select_first_frame_of_video(files_path):
    
    #Create a ROIS and ROI frames folder
    try:
        os.mkdir(os.path.join(files_path, "ROIS"))
        os.mkdir(os.path.join(files_path, "ROI_Frames"))
    except:
        pass

    match = re.search(r"([0-9]+)", files_path)
    session_code=match.group(1)
    
    # Only select and save frames from videos if the directory frame path is empty
    frame_path = os.path.join(files_path, "ROI_Frames")
    
    if not os.listdir(frame_path):    
        print("Directory is empty. Selecting the 1st frame from the videos")
        select_and_save_frame_from_videos_in_path(files_path)   
    else:    
        print("Directory is not empty. No selection of frames from videos")  

        
def convert_px_to_cm(df, session_code, filename, path):
    
    cols_to_convert=['x', 'y','width', 'height']
    # Convert df data from px to cm
    df[cols_to_convert] = (df[cols_to_convert]*10)/50
    
    #Save df
    file_path=os.path.join(path, "ROIS", "%s_%s.csv"%(session_code, filename))    
    df.to_csv(file_path, header=True)
    print(df.head())
    


def collect_maze_limits(path):
   
    # Extract first frame of the video in path
    select_first_frame_of_video(path)
    
    # Collect start limit from frame
    start_roi_px, session_code = collect_rois_df(path, 'start_roi_px')
    
    # Collect CP limits from frame
    cp_roi_px, session_code = collect_rois_df(path, 'cp_roi_px')
    
     # Collect Corner 1 limits from frame
    corner1_roi_px, session_code = collect_rois_df(path, 'corner1_roi_px')
    
    # Collect RW1 limits from frame
    rw1_roi_px, session_code = collect_rois_df(path, 'rw1_roi_px')
    
     # Collect Corner 2 limits from frame
    corner2_roi_px, session_code = collect_rois_df(path, 'corner2_roi_px')
    
    # Collect RW2 limits from frame
    rw2_roi_px, session_code = collect_rois_df(path, 'rw2_roi_px')
    
    
    # Convert measurments from pixels (px) to cm
    convert_px_to_cm(start_roi_px, session_code, 'start_roi_converted', path)
    convert_px_to_cm(cp_roi_px, session_code, 'cp_roi_converted', path)
    convert_px_to_cm(rw1_roi_px, session_code, 'rw1_roi_converted', path)
    convert_px_to_cm(rw2_roi_px, session_code, 'rw2_roi_converted', path)
    convert_px_to_cm(corner1_roi_px, session_code, 'corner1_roi_converted', path)
    convert_px_to_cm(corner2_roi_px, session_code, 'corner2_roi_converted', path)
    
    return session_code


# In[ ]:
#--------------------------- ADD RUN SPECIFICATIONS ----------------------#

def add_run_specs(files_path, session_code):
    
    # Open raw data from step2 
    data_path = os.path.join(files_path,"Timestamped_position","%s_timestamped_position_df_raw.csv"%session_code)
    df1 = pd.read_csv(data_path, header=0, index_col=0)
    
    # Add three new columns to df
    df2 = pd.DataFrame(columns=['run_type', 'outcome', 'trial_nr'])
    df = pd.concat([df1, df2], axis=1)
    
    # Collect list of file names within files_path containing the given string pattern
    run_specs_files = pd.Series(get_file_list(os.path.join(files_path, 'Run_Specs'), "*.csv"))
    specs_path = os.path.join(files_path, 'Run_Specs', run_specs_files[0])

    # Collect number of position points per each run
    points_per_run = (df.loc[:, 'run_nr']).value_counts().sort_index()

    try:
            run_specs = pd.read_csv(specs_path, header=None, delimiter=',') 
    except:
            run_specs = pd.read_csv(specs_path, header=None, delimiter=';')

    df.loc[:, 'run_type'] = run_specs[1].repeat(points_per_run).tolist()
    df.loc[:, 'outcome'] = run_specs[2].repeat(points_per_run).tolist()
    df.loc[:, 'trial_nr'] = run_specs[3].repeat(points_per_run).tolist()
    df = df.dropna(axis=0)
    
    return df


def classify_maze_segments(path, data, session_code):
    
    '''
    Classify xy datapoints according to maze segment using ROI limits
    '''
    # Open roi files
    start_roi = pd.read_csv(os.path.join(
        path, "ROIS", '{}_start_roi_converted.csv'.format(session_code)))
    cp_roi = pd.read_csv(os.path.join(
        path, "ROIS", '{}_cp_roi_converted.csv'.format(session_code)))
    corner1_roi = pd.read_csv(os.path.join(
        path, "ROIS", '{}_corner1_roi_converted.csv'.format(session_code)))
    corner2_roi = pd.read_csv(os.path.join(
        path, "ROIS", '{}_corner2_roi_converted.csv'.format(session_code)))
    
    # Get limits
    start_x = start_roi['x'].iloc[0]
    cp_x = cp_roi['x'].iloc[0]
    cp_y = cp_roi['y'].iloc[0]
    corner1_y = corner1_roi['y'].iloc[0]
    corner2_y = corner2_roi['y'].iloc[0]
    
    # In start_roi
    data.loc[data['x']<start_x, 'maze_segment'] = 'Start arm'
    
    # Central arm / Reward arms
    data.loc[data['x'].between(start_x, cp_x), 'maze_segment'] = 'Central/Rw arms'
    
    # Choice point
    data.loc[(data['y'].between(cp_y-10,cp_y+20)) & (data['x']>=cp_x), 'maze_segment'] = 'Choice point'
    
    # Corners
    data.loc[
        (data['x']>=cp_x) & (data['y']>=corner2_y), 'maze_segment'] = 'Corners'
    
    data.loc[
        (data['x']>=cp_x) & (data['y']<=corner1_y+10), 'maze_segment'] = 'Corners'
    
    
    # Choice point to corners
    data['maze_segment'] = data['maze_segment'].fillna('Pre-corners') 
    
    return data

