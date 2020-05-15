#ignoring deprecation and convergence warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import sys
import os
from helper import get_meal_vectors
from pathlib import Path
from dynaconf import settings
import pandas as pd
import numpy as np
import helper
from sklearn.externals import joblib
from features.features import generate_features 
import argparse

if __name__ == "__main__":

    #error for positional args file_name
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    args = parser.parse_args()

    #reading file and converting to vector
    filename = args.file_name
    meal_data_np = []
    print("loading file - " + filename)
    meal_data = pd.read_csv(os.path.join(filename), na_filter = False, header = None, sep = '\n')
    for i,_ in enumerate(meal_data.iterrows()):
            t = helper.getFloatFromObjectForMealData(meal_data.loc[i])
            if t.size != 0: 
                t = t[::-1]
                meal_data_np.append(t)                
    meal_data_np = np.array(meal_data_np)

    #reading all models and their settings
    directory = Path(settings.path_for(settings.FILES.MODELS))
    directory = str(directory)
    model_dict = list(settings.CLASSIFIER.MODEL_DICT)

    #load from saved models and run predict for generated vectors
    classifier_preditions = pd.DataFrame()
    for classifier in model_dict:
        filename = classifier[1]
        model = joblib.load(os.path.join(directory, filename))
        meal_vectors, labels = get_meal_vectors(classifier[0], apply_pca=True, padding=False, load_pca=True)
        predictions = model.predict(meal_vectors)
        classifier_preditions[classifier[0]] = predictions
    
    # output_folder = Path(settings.path_for(settings.FILES.OUTPUT_DIRECTORY))
    output_file = 'classifier_predictions.csv' 
    classifier_preditions.to_csv(output_file, index=False)
    print("###################################################")
    print("Check {} for classifier outputs".format(output_file))
    print("###################################################")