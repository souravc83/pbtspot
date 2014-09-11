#import std modules
import numpy as np
import scipy.ndimage as ndimage
#import local modules


class CurrImage:
    #should this get filename or the image np array itself
    #I think np array
    def __init__(self,imgnp):
        self.image=imgnp;
        [self.xlim,self.ylim]=self.image.shape;
        
        #to be defined later
        self.int_image=None;
        self.rotated_int_image=None;
    
    
    #zero mean etc? need a function for that   
    
    def calc_integral_image(self):
        self.int_image=np.zeros((self.xlim,self.ylim));
        
        for x_val in range(self.xlim):
            s_xy=0;
            
            for y_val in range(self.ylim):
                s_xy=s_xy+self.image[y_val,x_val];
                if x_val==0:
                    self.int_image[y_val,x_val]=s_xy;
                else:
                    self.int_image[y_val,x_val]=s_xy+self.int_image[y_val,x_val-1];
        return;
        
    
    def get_rsat(self,x_val,y_val):
        if(x_val>=0 and y_val>=0):
            return self.rotated_int_image[y_val,x_val];
        else:
            return 0;
            
    
    def calc_rotated_integral_image(self):
        self.rotated_int_image=np.zeros(self.xlim,self.ylim);
        
        #First Pass,left to right, top to bottom
        for x_val in range(self.xlim):
            for y_val in range(self.ylim):
                self.rotated_int_image=self.get_rsat(x_val-1,y_val-1)+\
                                       self.get_rsat(x_val-1,y_val)+\
                                       self.image[y_val,x_val]+\
                                       self.get_rsat(x_val-2,y_val-1);
        
        #Second Pass,right to left, bottom to top                        
        for x_val in range(self.xlim-1,-1,-1):
            for y_val in range(self.ylim-1,-1,1):
                self.rotated_int_image=self.get_rsat(x_val,y_val)+\
                                       self.get_rsat(x_val-1,y_val+1)+\
                                       self.get_rsat(x_val-2,y_val);
                                       
                                       
                                       
        return;

                                                              