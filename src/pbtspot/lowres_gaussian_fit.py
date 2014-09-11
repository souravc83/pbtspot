#global module imports
from __future__ import division
import numpy as np
import math

#local module imports
import train_and_test, globalconstants
from adaboost import example_images,example_lowres
from adaboost.generateimage import image_generate,single_lowres

class LowresGaussianFit(train_and_test.TrainAndTest):
    """
    This class trains a boosting tree for lowres images
    and tests for a single image
    
    public methods:
    generate_matrix()
    
    public variables:
    self.prob_matrix
    self.matrix_gaussianfit
    
    """
    
    def __init__(self,SNR):
        train_and_test.TrainAndTest.__init__(self,SNR);
        
        glob_const=globalconstants.GlobalConstants();
        self.global_window_width=glob_const.global_window_width;
        self.global_window_height=glob_const.global_window_height;
        
        
        self.min_x= -5;
        self.max_x= 5;
        self.min_y= -5;
        self.max_y= 5;
        
        self.region_x=range(self.min_x,self.max_x+1);
        self.region_y=range(self.min_y,self.max_y+1);
        
        #to be defined later
        self.prob_matrix=None;
        self.matrix_gaussianfit=None;
        
    
    def _generate_samples(self):
        
        
        [xleft,xright,ytop,ybottom]=self._get_window_parameters();
        test_img=single_lowres.SingleLowres(self.SNR);
        test_img.lowres_image();
        [particle_x,particle_y]=test_img.particle_loc[0];
        
        particle_x=int(particle_x/4);
        particle_y=int(particle_y/4);
        
        
        
        
        
        for y_val in self.region_y:
            center_y=particle_y+y_val;
            for x_val in self.region_x:
                center_x=particle_x+x_val;
                thisimg=test_img.scaled_noisy_lowres[center_y-ytop:center_y+ybottom,\
                                                 center_x-xleft:center_x+xright];
                if y_val ==0 and x_val==0:
                    thisexample=example_images.Example(thisimg,True,1.);
                else:
                    thisexample=example_images.Example(thisimg,False,1.);
                
                self.prob_matrix[(y_val+self.min_y),(x_val+self.min_x)]=\
                                self._test_one_example(thisexample);
        return;
    
    
    def generate_matrix(self):
        self.prob_matrix=np.zeros((len(self.region_y),len(self.region_x)),dtype=np.float);
        self._generate_samples();
        
    def fit_gaussian(self):
        if self.prob_matrix==None:
            raise AttributeError("prob_matrix not defined");
    
    
    def _get_window_parameters(self):
        """
        returns a list:[xleft,xright,ytop,ybottom]
        """
        xleft=int(math.ceil(self.global_window_width/2));
        xright=int(math.floor(self.global_window_width/2))+1;
        ytop=int(math.ceil(self.global_window_height/2));
        ybottom=int(math.floor(self.global_window_height/2))+1;
        return [xleft,xright,ytop,ybottom];
        
                
                
                
                
                
        
        
    
    
        
