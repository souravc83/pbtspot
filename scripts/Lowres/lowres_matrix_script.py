"""
This script generates a matrix of probability values
it also generates a Gaussian fit for it

"""
#global module imports
from __future__ import division
import matplotlib.pyplot as plt

#local module imports
from pbtspot import lowres_gaussian_fit


def matrix_script():
    #init_set ...
    SNRval=2.;
    lowgauss=lowres_gaussian_fit.LowresGaussianFit(SNRval);
    lowgauss.train_tree();
    lowgauss.generate_matrix();
    