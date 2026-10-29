import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skopt
from FluxFox.ameriflux import retrieve_ameriflux
from FluxFox.postproc import ustar_papale_2006, despike_mad_papale_2006, gapfill_xgb_liu_2025, compute_isday, gpp_night_reichstein_2005
from FluxFox.postproc.partition import ReichsteinResult

username = "afox18"  # change me to your ameriflux username
email = "afox18@uwyo.edu"  # change me to the email associated with your ameriflux account