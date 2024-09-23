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
     
