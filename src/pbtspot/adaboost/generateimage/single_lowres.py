"""
@author: Sourav Chatterjee
@brief: Generates synthetic noisy images with particles
"""

#import from std modules
from __future__ import division
import numpy as np
import math

#import from local modules
import image_generate


class SingleLowres(image_generate.GenerateImage):
    
    def __init__(self,SNRval):
        image_generate.GenerateImage.__init__(self,SNRval);
    
    def generate_particles(self):
        
        bound=6*max(self.sigma_x,self.sigma_y)
        
        #so we can ensure atleast a 10*10 region
        if bound<=40:
            bound=40;
        
        
        x_loc=bound+(self.width-2*bound)*np.random.rand();
        y_loc=bound+(self.height-2*bound)*np.random.rand();
        
        x_loc=math.floor(x_loc);
        y_loc=math.floor(y_loc)
        
        if self.particle_shape =='elongated':
            ran_theta=2*math.pi*np.random.random_sample(size=1);
        elif self.particle_shape =='round':
            ran_theta=0.
        else:
            raise ValueError('particle shape is round or elongated')
            
        #Initializes self.part_arr containing particle intensities
        self.part_arr=np.zeros((self.height,self.width));
        self.particle_loc=[];
        self.particle_loc.append([x_loc,y_loc]);
        noise_loc=self.background_level+ self.bg_noise[y_loc,x_loc];
        particle_intensity=self.SNR*self.SNR+self.SNR*math.sqrt(self.SNR*self.SNR + 4.0*noise_loc);
        self.part_arr[y_loc,x_loc]=particle_intensity;
        self._particle_gaussian(x_loc,y_loc,ran_theta);
        return;
    
    
    
    
    