import pandas as pd
from tsfresh.feature_extraction.feature_calculators import \
standard_deviation, fft_aggregated,  longest_strike_above_mean, linear_trend, \
count_above_mean, count_below_mean, time_reversal_asymmetry_statistic, skewness, variance, mean, median, \
mean_change, maximum, abs_energy, absolute_sum_of_changes, kurtosis, ratio_beyond_r_sigma, mean_abs_change
import numpy as np
from dynaconf import settings
from ast import literal_eval
from features.get_pca import get_pca_vectors


def get_range_in_windows(arrayOrg):
    """get ranges for dataframe to perform min max computation
    
    Arguments:
        arrayOrg {Pandas.Series} -- group chunk to get min max on
    
    Returns:
        float -- max value across windows
    """
    windowSize = int(arrayOrg.shape[0] / 5)
    if windowSize < 5: return np.max(arrayOrg) - np.min(arrayOrg)

    arrayLength = arrayOrg.shape[0]
    lastValue = arrayOrg[arrayLength - 1]
    arrayPadded = np.pad(arrayOrg, ((0, windowSize - 1)), mode='constant', constant_values=lastValue)
    rangeArray = []
    for index in range(arrayLength):
        window = arrayPadded[index: index + windowSize]
        minMinusMax = np.max(window) - np.min(window)
        rangeArray.append(minMinusMax/windowSize)

    np_max = np.max(np.array(rangeArray))
    return np_max

def get_min_max(df):
    """gets min max of dataframe in windows
    
    Arguments:
        df {Pandas.DataFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with min max column for each patient_number, meal_number combination
    """
    mealGroups = df.dropna(subset=['cgm_data']).groupby(['patient_number', 'meal_number']).apply(lambda group: get_range_in_windows(group.cgm_data.to_numpy()))
    return mealGroups.reset_index().rename(columns={0: 'min_max'})


def get_min_max_for_npd(df):
    """gets min max of dataframe in windows

    Arguments:
        df {Pandas.DataFrame} -- patient dataframe

    Returns:
        Pandas.DataFrame -- dataframe with min max column for each patient_number, meal_number combination
    """
    windows = np.array(get_range_in_windows(df))
    res = np.array([windows])
    res = np.nan_to_num(res)
    return res

def get_fft(arr):
    attrs = [
        {'aggtype': 'centroid'},
        {'aggtype': 'variance'},
        {'aggtype': 'skew'},
        {'aggtype': 'kurtosis'},
    ]
    
    fft_zip = fft_aggregated(arr, attrs)
    res = np.array([item[1] for item in list(fft_zip)])
    res = np.nan_to_num(res)
    
    return res

def get_ratio_beyond_r_sigma(arr):
    res = np.array([ratio_beyond_r_sigma(arr)])
    res = np.nan_to_num(res)
    return res

def get_kurtosis(arr):
    res = np.array([kurtosis(arr)])
    res = np.nan_to_num(res)
    return res

def get_absolute_sum_of_changes(arr):
    res = np.array([absolute_sum_of_changes(arr)])
    res = np.nan_to_num(res)
    return res

def get_abs_energy(arr):
    res = np.array([abs_energy(arr)])
    res = np.nan_to_num(res)
    return res

def get_maximum(arr):
    res = np.array([maximum(arr)])
    res = np.nan_to_num(res)
    return res

def get_median(arr):
    res = np.array([median(arr)])
    res = np.nan_to_num(res)
    return res


def get_variance(arr):
    res = np.array([variance(arr)])
    res = np.nan_to_num(res)
    return res


def get_mean(arr):
    res = np.array([mean(arr)])
    res = np.nan_to_num(res)
    return res

def get_sd(arr):
    res = np.array([standard_deviation(arr)])
    res = np.nan_to_num(res)
    return res

def get_cam(arr):
    res = np.array([count_above_mean(arr)])
    res = np.nan_to_num(res)
    return res

def get_cbm(arr):
    res = np.array([count_below_mean(arr)])
    res = np.nan_to_num(res)
    return res

def get_mean_change(arr):
    res = np.array([mean_change(arr)])
    res = np.nan_to_num(res)
    return res


def get_rms(arr):
    res = np.array([np.sqrt(np.mean((arr)**2))])
    res = np.nan_to_num(res)
    return res


def get_lsam(arr):
    res = np.array([longest_strike_above_mean(arr)])
    res = np.nan_to_num(res)
    return res


def get_lt(arr):
    params = [
        {'attr': 'pvalue'},
        {'attr': 'rvalue'},
        {'attr': 'slope'},
        {'attr': 'stderr'}
    ]
    res = np.array([item[1] for item in linear_trend(arr, params)])
    return np.nan_to_num(res)


def get_time_reversal(arr):
    res = np.array([time_reversal_asymmetry_statistic(arr, lag=1)])
    res = np.nan_to_num(res)
    return res

def get_skew(arr):
    res = np.array([skewness(arr)])
    res = np.nan_to_num(res)
    return res

def get_mean_abs_change(arr):
    res = np.array([mean_abs_change(arr)])
    res = np.nan_to_num(res)
    return res

def get_feature_func(feature_name):
    """Dynamically maps feature name to functions
    
    Arguments:
        feature_name {str} -- name of feture
    
    Returns:
        function -- corresponding feature function
    """
    feature_to_function_map = {
        'linear_trend': get_lt,
        'rms': get_rms,
        'min_max': get_min_max_for_npd,
        'lsam': get_lsam,
        'fft': get_fft,
        'cam': get_cam,
        'cbm': get_cbm,
        'time': get_time_reversal,
        'skew': get_skew,
        'variance': get_variance,
        'mean': get_mean,
        'median': get_median,
        'sd': get_sd,
        'mean_change': get_mean_change,
        'maximum': get_maximum,
        'abs_energy':get_abs_energy,
        'absolute_sum_of_changes': get_absolute_sum_of_changes,
        'kurtosis': get_kurtosis,
        'ratio_beyond_r_sigma': get_ratio_beyond_r_sigma,
        'mean_abs_change': get_mean_abs_change
    }

    return feature_to_function_map[feature_name]


def generate_features(model_name, meal_array, apply_pca=True, load_pca = False):
    """generates features from meal array data
    
    Arguments:
        meal_array {np.array} -- 2D numpy array of meal and no meal data
    
    Keyword Arguments:
        apply_pca {bool} -- set to False if data should not be reduced (default: {True})
    
    Returns:
        np.array -- 2D numpy array of meal data features
    """
    print("using {} for predicting".format(model_name))
    feature_array = np.array([])

    if model_name == "randomForestClassifier":
        k = int(settings.FEATURES_RFC.K)
        features = list(settings.FEATURES_RFC.FEATURES)
    if model_name == "decisionTreeClassifier":
        k = int(settings.FEATURES_DTC.K)
        features = list(settings.FEATURES_DTC.FEATURES)
    if model_name == "multiLayerPerceptronClassifier":
        k = int(settings.FEATURES_MLP.K)
        features = list(settings.FEATURES_MLP.FEATURES)
    if model_name == "supportVectorMachineClassifier":
        k = int(settings.FEATURES_SVM.K)
        features = list(settings.FEATURES_SVM.FEATURES)
    
    for feature in features:
        res = map(get_feature_func(feature), meal_array)
        res = np.array(list(res))
        # print("Feature: {} | Size: {}".format(feature, res.shape))

        if not feature_array.size: feature_array = res
        else: feature_array = np.concatenate((feature_array, res), axis=1)

    if apply_pca:
        feature_array = get_pca_vectors(model_name, feature_array, k, load_pca)

    return feature_array
