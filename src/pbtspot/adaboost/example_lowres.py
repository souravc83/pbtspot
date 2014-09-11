#global module imports
from __future__ import division
import math
import random

#local module imports
import example_images
from pbtspot import globalconstants
from generateimage import image_generate

class LowresImages(example_images.ExampleImages):
    """
    This class creates lowres images and behaves 
    exactly with the same interface as its highres
    version
    """
    def __init__(self,SNR,no_pos_eg,no_neg_eg):

        example_images.ExampleImages.__init__(self,SNR,no_pos_eg,no_neg_eg); 
        glob_const=globalconstants.GlobalConstants();
        self._lowres_width=glob_const.lowres_width;
        self._lowres_height=glob_const.lowres_height;
        self.img_width=512;
        self.img_height=512;
        
        self._ratio=None;
        self._set_ratio();



    def _set_ratio(self):
        """
        @defines self._ratio
        """
        if self.img_width % self._lowres_width != 0:
            raise ValueError("High Res/low res should be whole number");

        if (self._img_width/self._lowres_width)!=(self._img_height/self._lowres_height):
            raise ValueError("Lowres width and height\
                              should have same scaling factor")

        self._ratio=self.img_width/self._lowres_width;
        return;

    def _gen_pos_examples(self):
        """
        overrides _gen_pos_examples in parent class. 
         generates all positive examples by generating a new image
        @writes to: self.training_examples: list of instances of class Example
        """
        if self._num_pos_eg ==0:
            raise ValueError("Number of positive examples is 0");
        self._img_gen.reset_image();
        self._img_gen.lowres_image();
        [xleft,xright,ytop,ybottom]=self._get_window_parameters();
        
        tot_images=self.num_particles;
        counter=0;
        while True:
            if counter == tot_images:
                self._img_gen.reset_image();
                self._img_gen.lowres_image();
            for location in self._img_gen.particle_loc:
                [x_loc,y_loc]=location;
                x_lowres=math.floor(x_loc/self._ratio);
                y_lowres=math.floor(y_loc/self._ratio);
                
                if self._check_coord(x_lowres-xleft,y_lowres-ytop) and\
                   self._check_coord(x_lowres+xright,y_lowres+ybottom):
                    #pos_img=self._img_gen.noisy_imgarr[y_loc-ytop:y_loc+ybottom,\
                    #                                 x_loc-xleft:x_loc+xright];
                    pos_img=self._img_gen.scaled_noisy_lowres[y_lowres-ytop:y_lowres+ybottom,\
                                                     x_lowres-xleft:x_lowres+xright];
                    thisexample=example_images.Example(pos_img,True,self._pos_example_weights);
                    
                    counter+=1
                    yield thisexample;
        return;


    def _gen_neg_examples(self):
        """
        overrides _gen_pos_examples in parent class. 
         generates all positive examples by generating a new image
        @writes to: self.training_examples: list of instances of class Example
        """
        if self._num_neg_eg ==0:
            raise ValueError("Number of positive examples is 0");
        self._img_gen.reset_image();
        self._img_gen.lowres_image();
        [xleft,xright,ytop,ybottom]=self._get_window_parameters();
        
        
        counter=0;
        while True:
            if counter == self.num_neg_eg_img:
                self._img_gen.reset_image();
                self._img_gen.lowres_image();
                
            #get random rows and columns
            rowno=random.randint(ytop,self._img_gen.lowres_height-ybottom)*self._ratio;
            colno=random.randint(xleft,self._img_gen.lowres_width-xright)*self._ratio;
            lowres_colno=math.floor(colno/self._ratio)
            lowres_rowno=math.floor(rowno/self._ratio)
            #Check if this is particle
            if [colno,rowno] in self._img_gen.particle_loc :
                continue;
                
            elif self._check_coord(lowres_colno-xleft,lowres_rowno-ytop)==False or\
                 self._check_coord(lowres_colno+xright,lowres_rowno+ybottom)==False:
                continue;
                    
            else:
                #neg_img=self._img_gen.noisy_imgarr[rowno-ytop:rowno+ybottom,\
                #                             colno-xleft:colno+xright];
                neg_img=self._img_gen.scaled_noisy_lowres[lowres_rowno-ytop:lowres_rowno+ybottom,\
                                             lowres_colno-xleft:lowres_colno+xright];
                thisexample=example_images.Example(neg_img,False,self._neg_example_weights);
                counter+=1;
                yield thisexample;          
        return;


    def _check_coord(self,x_loc,y_loc):
        width=self._img_gen.lowres_width;
        height=self._img_gen.lowres_height;
        
        if x_loc>=0 and x_loc<width and \
           y_loc>=0 and y_loc<height:
            return True;
        else:
            return False;
        
    
   
        

        



