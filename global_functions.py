# Copyright 2019 Max Planck Society. All rights reserved.

import numpy as np

def rmse(predictions, targets):
    assert predictions.ndim == targets.ndim == 1
    return np.sqrt(((predictions - targets) ** 2).mean())

def rms(predictions):
    assert predictions.ndim == 1
    return np.sqrt(((predictions) ** 2).mean())

def r_squared(Y_GP, Y_train):
    """
    Computes percentage of variance explained
    input Y_train: Observations
    input Y_GP: GP mean
    variable ybar: Empirical mean of observations
    """
    assert Y_train.ndim == Y_GP.ndim == 1
    ybar = Y_train.mean()          # or sum(y)/len(y)
    ssreg = np.sum((Y_GP - ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    #ssres = np.sum((Y_GP - Y_train)**2)
    sstot = np.sum((Y_train - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    #return 100*(1-(ssres/sstot))
    return 100 * (ssreg / sstot)