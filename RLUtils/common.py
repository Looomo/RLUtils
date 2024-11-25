import glob
import os
import torch
import numpy as np
import pickle as pkl
import random
import time
import string
from .mylogger import MyLogger
import time
import copy


def cal_distance(a, b, metric = "Chebyshev"):

    if metric == "Chebyshev":
        distance = np.max( np.abs( a - b ) )
    elif metric == "Euclidean":
        distance = np.linalg.norm(a - b )
    elif metric == "Cosine":
        distance = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        distance = 1 - (distance + 1)/2
    else:
        raise NotImplementedError(f"Distance metric {metric} not supported yet.")

    return distance

def check_undetected_terminal(dataset, thresh = 1.0, metric = "Chebyshev"):
    for i in range( len(dataset['observations']) - 1):
        if cal_distance(dataset['observations'][i+1], dataset['observations'][i], metric=metric ) > thresh:
            assert dataset['terminals'][i], f"Found undetected terminal: {i}"

def check_invalid_terminal(dataset, thresh = 1.0, metric = "Chebyshev"):
    # for i in range( len(dataset['observations']) - 1):
    #     if np.max( np.abs( dataset['observations'][i+1] - dataset['observations'][i] ) ) > thresh:
    #         assert dataset['terminals'][i], f"Found undetected terminal: {i}"

    for i in range( len(dataset['observations']) - 1):
        if dataset['terminals'][i]:
            assert  cal_distance(dataset['observations'][i+1], dataset['observations'][i], metric=metric ) > thresh, f"Found invalid terminal: {i}"
    return


def rebuild_terminals_by_thresh(dataset_, thresh = 1.0 , metrics = ""):
    dataset = copy.deepcopy(dataset_)
    dataset['terminals'][:] = 0.

    for i in range(len(dataset['terminals']) - 1):
        if np.max( np.abs( dataset['observations'][i + 1] - dataset['observations'][i] ) ) > thresh:
            dataset['terminals'][i] = 1.
        else:
            dataset['terminals'][i] = 0.

    return dataset

def get_latest_epoch(loadpath, template = 'state_[0-9]*.pt', extend_with_point = '.pt'):
    states = glob.glob1(loadpath, template)
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace(extend_with_point, ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch


def seed_torch(seed=42, deterministic = False, externel_logger:MyLogger = None):

    message = f"Setting seed to: {seed}, deterministic = {deterministic}"
    if externel_logger is not None:
        externel_logger.info( message )
    else:
        print(message)
    random.seed(seed)   
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)  
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    if deterministic:
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True 
    
    message = f"Seed set."
    if externel_logger is not None:
        externel_logger.info( message )
    else:
        print(message)

def get_token():
    current_timestamp = time.time()
    state = random.getstate()
    random.seed(current_timestamp)
    letters = string.ascii_letters + string.digits
    token = ''.join(random.choice(letters) for _ in range(16))
    random.setstate(state)
    return token



def sort_low_to_high(to_sort, values):
    inds = np.argsort(values)[::-1]
    if to_sort is not None:
        to_sort = to_sort[inds]
    values = values[inds]
    return to_sort, values



class Timer:
	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff
     
