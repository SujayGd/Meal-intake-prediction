import pandas as pd
from configparser import ConfigParser
import ast
from helper import convert_to_epoch

k = ConfigParser()
k.read('config.ini')
file_map = ast.literal_eval(k['FILES']['CGM_files'])
directory = k['FILES']['data_directory']

patient_df = pd.DataFrame()

for patient_number, files in file_map.items():
    time_frame = pd.read_csv(directory + files['time_series'], na_filter=False)
    cgm_frame = pd.read_csv(directory + files['data'], na_filter=False)

    #transpose CSVs and join each column btwn datacsv and timecsv to a dataframe 
    #with meal and patient number
    for meal_number in range(cgm_frame.T.shape[1]):
        cgm_data = cgm_frame.T.iloc[:, meal_number].values
        time_data = time_frame.T.iloc[:, meal_number].values
        meal_data_frame = pd.DataFrame({
            'meal': meal_number,
            'patient': int(patient_number[-1]),
            'cgm_data': cgm_data,
            'time_data': time_data
        })
        patient_df = patient_df.append(meal_data_frame)

#convert timeseries to python datetime
patient_df['time_data'] = patient_df['time_data'].apply(lambda cell: convert_to_epoch(cell))
patient_df.to_csv('test.csv', index=False)





