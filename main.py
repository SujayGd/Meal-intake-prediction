import pandas as pd
from configparser import ConfigParser
import ast
from helper import convert_to_epoch

#read config file and get patient data sources
k = ConfigParser()
k.read('config.ini')
file_map = ast.literal_eval(k['FILES']['CGM_files'])
directory = k['FILES']['data_directory']

patient_df = pd.DataFrame()
for patient_number, files in file_map.items():

    #read dataframes and convert each row to numpy arrays
    time_frame = pd.read_csv(directory + files['time_series'], na_filter=False)
    cgm_frame = pd.read_csv(directory + files['data'], na_filter=False)
    time_frame_array = time_frame.to_numpy()
    cgm_frame_array = cgm_frame.to_numpy()

    #zip functions joins each ith element of 2 arrays together:
    #zip([a1,a2],[b1,b2]) = [(a1,b1), (a2,b2)]
    #enumerate fetches index for each element in zip list.
    for index, (cgm_data,time_data) in enumerate(zip(cgm_frame_array, time_frame_array)):
        meal_data_frame = pd.DataFrame({
            'meal_number': index,
            'patient_number': int(patient_number[-1]),
            'cgm_data': cgm_data,
            'time_data': time_data
        })
        patient_df = patient_df.append(meal_data_frame)

#convert timeseries to python datetime and save to CSV *check output
patient_df['time_data'] = patient_df['time_data'].apply(lambda cell: convert_to_epoch(cell))
patient_df.to_csv('test.csv', index=False)





