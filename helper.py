from datetime import datetime, timedelta
import numpy as np

def convert_to_epoch(matlab_datenum):
    """converts shitty matlab time to Python datetime object
    
    Arguments:
        matlab_datenum {float} -- Matlab time with floating point
    
    Returns:
        datetime -- Python datetime object 
    """
    try:
        return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    except Exception:
        return 'Nan'
        # return np.nan
