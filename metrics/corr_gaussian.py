# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as tn
import torchtt as tntt
import numpy as np
import matplotlib.pyplot as plt
from src.qtt_interpolation.utils import *
from src.qtt_interpolation.int_tools import *

# %% Configuration

''' 
# Create a random 2D correlation matrix
A = tn.randn(2, 2)
cov = A @ A.t()  
D = tn.diag(1 / tn.sqrt(tn.diag(cov)))
corr = D @ cov @ D  
'''
#Midcorrelated variables
corr = np.array([[1.0, 0.5],
                         [0.5, 1.0]]) 
# %% Coarse grid

# %%Fine grid

# %% Collect data

# %% Save results
