#std imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#local imports
import currimage
from pbtspot import globalconstants

class Rectangle:
    """
    defines a rectangle in Haar feature 
    @param: x_val,y_val,width,height:integer>0
    @param: weight: positive or negative integer    
    """
    def __init__(self,x_val,y_val,width,height,weight):
        self.check_values(x_val,y_val,width,height,weight);
        

        
        #also check whether all four points within image limits, make a function
        #if you want weights to be +-1, check that too.
        self.x_val=x_val;
        self.y_val=y_val;
        self.width=width;
        self.height=height;
        self.weight=weight;
    
    def __str__(self):
        return("Rectange.X:%d,Y:%d,W:%d,H:%d"%(self.x_val,self.y_val,self.width,self.height))
    
    def __repr__(self):
        return("Rectange.X:%d,Y:%d,W:%d,H:%d","Wt:%d"%(self.x_val,self.y_val,self.width,self.height,self.weight))
        
    def check_values(self,x_val,y_val,width,height,weight):
        
        #Todo: this class must be able to access image size variables
        if(width<0):
            raise ValueError("Width value must be within limit:%d",width);
            
        if(height<0):
            raise ValueError("Height value must be within limit:%d",height);
        
        if(x_val<0):
            raise ValueError("X value must be within limit:%d",x_val);
            
        if(y_val<0):
            raise ValueError("Y value must be within limit:%d",y_val);
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)




class BaseFeature:
    """
    defines a BaseFeature. Different Feature types derive from it.
    defines virtual functions.
    """
    #We define features simply as a list of rectangles
    def __init__(self):
        glob_const=globalconstants.GlobalConstants();
        self.global_window_width=glob_const.global_window_width;
        self.global_window_height=glob_const.global_window_height;
        
        
        self.rectlist=None; #defined by subclass
        
        #Other variables, to be defined during optimization
        #should we initialize them here?
        self.threshold=None;
        self.toggle=None;
        self.error=None
        self.margin=None;
        self.response_vector=None;
    
    
    #Abstract classes, to be defined later
    def generate_feature(self,x_val,y_val,width,height): 
        raise NotImplementedError('Must implement a "generate_feature" method.')
    
    def __repr__(self):
        raise NotImplementedError('Subclass must implement a "__repr__" method.')
    
    #Implemented Classes
    #this class now takes in int_image, not image
    #taking in image was the main cause of slowing down the program
    # for each feature it kept on calculating the same int_image
    def convolve_int_image(self,int_image):
        
        #int_image=image;
        #currimg=currimage.CurrImage(image);
        #currimg.calc_integral_image();
        #int_image=currimg.int_image;
        
        if self.rectlist is None:
            raise AttributeError("Rectlist not defined by sublass yet.")
            
        sum_px=0;
        #can optimize by caching some values
        for rect in self.rectlist:
            sum_px+=rect.weight*self.sum_int_image(int_image,rect);
        return sum_px;
        
        
    def sum_int_image(self,int_image,rect):
        top_left=self.int_image_val(int_image,rect.y_val-1,rect.x_val-1);
        bottom_left=self.int_image_val(int_image,rect.y_val+rect.height-1,rect.x_val-1);
        top_right=self.int_image_val(int_image,rect.y_val-1,rect.x_val+rect.width-1);
        bottom_right=self.int_image_val(int_image,rect.y_val+rect.height-1,rect.x_val+rect.width-1);
        
        #print rect
        #print (bottom_right+top_left-bottom_left-top_right)
        #From Figure 3 of Viola-Jones Paper
        return (bottom_right+top_left-bottom_left-top_right);
    
    def sum_rotated_int_image(self,rotated_int_image,rect):
        first_term=self.rotated_int_image_val(rotated_int_image,rect.y_val+rect.width,rect.x_val+rect.width);
        second_term=self.rotated_int_image_val(rotated_int_image,rect.x_val-rect.height,rect.y_val+rect.height);
        third_term=self.rotated_int_image_val(rotated_int_image,rect.x_val,rect.y_val);
        fourth_term=self.rotated_int_image_val(rotated_int_image,rect.x_val+rect.width-rect.height,\
                                               rect.y_val+rect.width+rect.height);
        
        
    
    def int_image_val(self,int_image, y_val,x_val):
        """
        this function returns the integral image coordinates
        and handles the special case when values are sought at the 
        margin of the matrix
        """
        if(x_val==-1 or y_val==-1):
            return 0;
        else:
            return int_image[y_val,x_val];
    
    def show_feature(self):
        """
        plots a feature in an image, for testing purposes
        """
        if self.rectlist==None:
            raise ValueError("Rectlist not defined yet");
        
        img_toshow=50*np.ones((self.global_window_height,self.global_window_width),dtype=int);#Grey
        for rect in self.rectlist:
            if(rect.weight>0):
                img_toshow[rect.y_val:rect.y_val+rect.height,\
                           rect.x_val:rect.x_val+rect.width]=100;#white
            else:
                img_toshow[rect.y_val:rect.y_val+rect.height,\
                           rect.x_val:rect.x_val+rect.width]=0;#black
        fig=plt.figure();
        ax=fig.add_subplot(111);
        ax.imshow(img_toshow,cmap=cm.Greys_r,interpolation='None');
        plt.colorbar
        plt.grid(True);
        plt.show()
        return
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)              
