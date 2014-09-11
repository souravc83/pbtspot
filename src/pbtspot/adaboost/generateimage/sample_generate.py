"""
@author: Sourav Chatterjee
@brief: generates sample test images which match a Haar feature
"""
#std module imports
from __future__ import division
import numpy as np

#local module imports
import image_generate
from pbtspot import globalconstants
from pbtspot.adaboost.features import basefeature

class TestImages(image_generate.GenerateImage):
    
    def __init__(self):
        glob_const=globalconstants.GlobalConstants()
        self.height=glob_const.global_window_height;
        self.width=glob_const.global_window_width;
        #image_generate.GenerateImage(11,11);
        self.gray_level=50;
        self.firstrect=None;
        self.secondrect=None;

    def gen_center_feature(self):
        """
        Generates a center feature with grey values around it
        """
        self.firstrect=basefeature.Rectangle(2,2,6,1,-1);
        self.imgarr=self.gray_level*np.ones((self.height,self.width),dtype=np.int);
        self.imgarr[2:5,2:8]=0;#big black spot
        self.imgarr[3:4,4:6]=100;#smaller white spot
        self.noisy_imgarr=self.imgarr;
        return;
    
    def gen_neg_blank(self):
        """
        Generates a negative example blank image with gray values
        """
        self.imgarr=self.gray_level*np.ones((self.height,self.width),dtype=np.int);
        self.noisy_imgarr=self.imgarr;
        return;
        
        
        
        
        
        