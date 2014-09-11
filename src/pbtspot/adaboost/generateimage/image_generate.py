"""
@author: Sourav Chatterjee
@brief: Generates synthetic noisy images with particles
"""

#import from std modules
from __future__ import division
import numpy as np
from scipy import misc
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

#import from local modules
from pbtspot import globalconstants


class GenerateImage:

    def __init__(self,SNR,width=512,height=512):
        """
        Generates a synthetic image of particles
        @param: SNR: float, signal to noise ratio
        @param width:integer, image width
        @param height: integer, image height

        public methods:

        def self.generate_img()
        def self.reset_image()
        def save_image(self,filename='A.jpg',highres=True):
        def lowres_image(self)

        public variables:
        self.noisy_imgarr
        self.particle_loc
        self.scaled_noisy_imgarr
        self.noisy_lowres
        self.scaled_noisy_lowres
        """ 
        if(height<=0):
            raise ValueError("Height must be positive:%d"%height);
        if(width<=0):
            raise ValueError("Width must be positive:%d"%width);
        
        #from global values
        glob_const=globalconstants.GlobalConstants();
        self.global_window_width=glob_const.global_window_width;
        self.global_window_height=glob_const.global_window_height;
        self.particle_shape=glob_const.particle_shape;#round or elongated        
        self.image_type=glob_const.image_type;#image type, determining the type of noise
        
        #To be defined later   
        self.part_arr=None;#image array containing only particle Gaussians
        self.imgarr=None;#image array with particles and background 
        self.noisy_imgarr=None;# noisy array, after adding Poisson noise to self.imgarr
        self.scaled_noisy_imgarr=None;#after scaling noisy_imgarr, so that mean=0, sttdev=1
        self.particle_loc=None;#array of particle coordinates
        self.bg_noise=None;
        
        #Only when required to generate low resolution images
        self.noisy_lowres=None;
        self.scaled_noisy_lowres=None;
        #consider putting these to global constants as well
        self.lowres_height=128;
        self.lowres_width=128;
        
        
        #from paper, resolution of camera is:
        #delta_x=50nm, delta_y=50nm. Thus all values are multiples of 
        #50nm. So, sigma_x=2 implies sigma_x=100nm, since 1 pixel=50nm
        
       
        #no. of particles: rows and columns
        
        self.part_rows=16;#by default
        self.part_columns=16;
        
        #std. dev for the Gaussian, in pixels
        self.sigma_sym=2.;
        self.sigma_major=4.;
        if self.particle_shape=='round':
            self.sigma_x=self.sigma_sym;
            self.sigma_y=self.sigma_sym;
        elif self.particle_shape=='elongated':
            self.sigma_x=self.sigma_major;
            self.sigma_y=self.sigma_sym;
 
        self.height=height;
        self.width=width;
        
        #Image parameters
        self.SNR=SNR;
        self.background_level=10;#noise background level
        
    
    def generate_img(self):
        """
        Main function which generates the images
        All other functions are called from here
        """
        self.imgarr=np.zeros((self.height,self.width),dtype=np.float);
        self.generate_noise();#does nothing for Type A images, adds gradient noise for Type B images
        self.generate_particles();
         
        #self.add_intensities();   
        self.imgarr=self.background_level+self.bg_noise+self.part_arr;
        self.noisy_imgarr=copy.deepcopy(self.imgarr);#this is very important.
        self._add_shot_noise(self.noisy_imgarr);
        self._normalize_image();
        return;
        
        
    def reset_image(self):
        """
        Resets all arrays to None, Use with caution
            
        """
        self.imgarr=None;
        self.noisy_imgarr=None;
        self.particle_x=None;
        self.particle_y=None;
        self.part_arr=None;
        self.scaled_imgarr=None;
        
        #Only when required to generate low resolution images
        self.noisy_lowres=None;
        self.scaled_noisy_lowres=None;
        return
        
    def show_image(self,img_type='Noisy'):
        """
        shows an noisy or denoised image
        @param: img_type: String, 'Noisy' or 'Denoised' or 'Lowres'
        """
        if(img_type!='Noisy' and img_type!='Denoised' and img_type!='Lowres'):
            print 'Image type should be Noisy or Denoised or Lowres';
            return;
        
        img_toshow=None;
    
        if(img_type=='Noisy'):
            img_toshow=self.noisy_imgarr;
        elif img_type=='Denoised':
            img_toshow=self.imgarr;
        else:
            if self.noisy_lowres==None:
                raise AttributeError("Lowres image not defined yet");
            else:
                img_toshow=self.noisy_lowres;
            

            
        fig=plt.figure();
        ax=fig.add_subplot(111);
        ax.imshow(img_toshow,cmap=cm.Greys_r,interpolation='nearest');
        numrows,numcols=img_toshow.shape;
    
        def format_coord(x,y):
            col=int(x);
            row=int(y);
        
            if col>0 and col<numcols and row>0 and row<numrows:
                z=img_toshow[row,col];
                return 'x=%1.1f, y=%1.1f, z=%1.1f'%(x, y, z);
            else:
                return 'x=%1.1f, y=%1.1f'%(x, y);
            
        ax.format_coord=format_coord;
        plt.show()
        return;
    
    def generate_particles(self):
        """
        Generates the particles
        defines: self.part_arr:np array of size height*width
        """
        #test if no_particles is integer>0,SNR>0
        rand_xloc=-0.5+0.5*np.random.random_sample(size=self.part_columns);
        rand_yloc=-0.5+0.5*np.random.random_sample(size=self.part_rows);
        
        if self.particle_shape =='elongated':
            ran_theta=2*math.pi*np.random.random_sample(size=(self.part_rows*self.part_columns));
        elif self.particle_shape =='round':
            ran_theta=[0.]*(self.part_rows*self.part_columns);
        else:
            raise ValueError('particle shape is round or elongated')
            
       
        shiftx=int(self.width/16);
        grid_width=int((self.width-2*shiftx)/self.part_columns);
       
       
        shifty=int(self.height/16);
        grid_height=int((self.height-2*shifty)/self.part_rows);
        
        #check whether given coordinates are a valid image
        if 2*shiftx<self.global_window_width or\
           2*shifty<self.global_window_height or\
           grid_width<self.global_window_width or\
           grid_height<self.global_window_height:
            raise ValueError("Size too small or particle numbers too large. Cannot Generate Particles/Image")
           
       
       
        grid_deviation_x=self.global_window_width;
        grid_deviation_y=self.global_window_height;
       
        #initialize particle location arrays
        particle_x=[];
        particle_y=[];
        self.particle_loc=[];
       
        for cols in range(self.part_columns):
            xloc=shiftx+grid_width*cols+int(grid_deviation_x*rand_xloc[cols]);
            if self._within_limits(xval=xloc):
                particle_x.append(xloc);
            else:
                raise ValueError('Particle X location out of bounds');
           
        for rows in range(self.part_rows):
            yloc=shifty+grid_height*rows+int(grid_deviation_y*rand_yloc[rows]);
            if self._within_limits(yval=yloc):
                particle_y.append(yloc);
            else:
                raise ValueError('Particle Y location out of bounds');
               
        #Initializes self.part_arr containing particle intensities
        self.part_arr=np.zeros((self.height,self.width));
   
        particle_counter=0;
        
        for rows in range(self.part_rows):
            y_loc=particle_y[rows];
            for cols in range(self.part_columns):
                x_loc=particle_x[cols];
                noise_loc=self.background_level+ self.bg_noise[y_loc,x_loc];
               
                particle_intensity=self.SNR*self.SNR+self.SNR*math.sqrt(self.SNR*self.SNR + 4.0*noise_loc);
                self.part_arr[y_loc,x_loc]=particle_intensity;
                self.particle_loc.append([x_loc,y_loc])
                #Convert from point intensity to Gaussian
                self._particle_gaussian(x_loc,y_loc,ran_theta[particle_counter]);
                particle_counter+=1;
        return;
        
    def save_image(self,filename='A.jpg',highres=True):
        """
        saves an image file, writing information from a numpy array
        @param filename: String, name of file
        @param higres:Boolean, if True, saves highres image, if false,
        saves low res image
        """
        if(highres==True and self.noisy_imgarr==None):
            self.generate_img();
        
        if(highres==False and self.noisy_lowres==None):
            if(self.imgarr==None):
                self.generate_img();
            self.lowres_img();
        #check for integer array <256
        if(highres==True):
            misc.imsave(filename,self.noisy_imgarr);
        else:
            misc.imsave(filename,self.noisy_lowres);
        return;
    
    def lowres_image(self):
        """
        Defines a Lower resolution image, given a higher resolution image
        by averaging over the pixels of the higher resolution image
        
        @defines: self.lowres: np array of floats
        @defines: self.noisy_lowres: np array of floats,which is the version with added shot noise
        
        """
        if(self.height%self.lowres_height)!=0 :
            raise ValueError("High Res Width not exact multiple of low res width")
        if(self.width%self.lowres_width)!=0 :
            raise ValueError("High Res Width not exact multiple of low res width")
        
        int_width=int(self.width/self.lowres_width);
        int_height=int(self.height/self.lowres_height);
        
        if self.imgarr==None:
            self.generate_img();
        
        #Initialize lowres image
        self.noisy_lowres=np.zeros((self.lowres_height,self.lowres_width),dtype=np.float);
        
        for col in range(self.lowres_width):
            col_start=int_width*col;
            col_end=col_start+int_width;
            
            for row in range(self.lowres_height):
                    row_start=int_height*row;
                    row_end=row_start+int_height;
                    
                    self.noisy_lowres[row,col]=np.mean(self.imgarr[row_start:row_end,col_start:col_end]);
        
        self._add_shot_noise(self.noisy_lowres);
        self._normalize_lowres();
        return;
    
    
    
    def _within_limits(self,xval=0,yval=0):
        """
        checks whether x and y locations are within bounds
        @retval: True if location is within bounds, otherwise false
        """    
        if(xval>=0 and xval<self.width and yval>=0 and yval<self.height):
            return True;
        else:
            return False;

        
    

    #Leave this for now, implemented only for Type B and Type C images
    def generate_noise(self):
        """
        Generates Gaussian noise
        @defines: bg_noise:np array size of image 
        """ 
        #the numerical values of 50 and 150 are just as in the paper
        
        self.bg_noise=np.zeros((self.height,self.width),dtype=np.float);
           
        if self.image_type == 'A':
            return;
        
        elif self.image_type =='B':
            for row in range(self.height):
                for col in range(self.width):
        		    self.bg_noise[row][col]=50.*float(col/(self.width-1));
            return;
        
        #elif self.image_type =='C': Do this later
        else:
            raise TypeError("Image Type has to be A or B")    
        

    def _gauss_2d(self,A0,x0,y0,sigma_x,sigma_y):
        """
        Calculates a simple 2D gaussian function, given the parameters
        """
        #does not use self.sigma_x and self.sigma_y, to make it more generalized
        exponent= ((x0**2)/(2*sigma_x**2)) + ((y0**2)/(2*sigma_y**2));
        return A0*np.exp(-exponent);

    def _particle_gaussian(self,x_loc,y_loc,ran_theta=0.):
        """
        Applies a Gaussian intensity profile on a single point
        @param: x_loc,y_loc: positive integers showing x and y locations in image
    
        """
        masksize_x=int(6*self.sigma_x)+1;#3 std. deviations 
        masksize_y=int(6*self.sigma_y)+1;
        
        if self._within_limits(x_loc,y_loc)==False or\
           self._within_limits(x_loc-masksize_x,y_loc-masksize_y)==False or\
           self._within_limits(x_loc+masksize_x,y_loc+masksize_y)==False:
            raise ValueError('Particle location passed to particle_gaussian not within limits,\
            x:%d,y:%d'%(x_loc,y_loc));
   
        particle_intensity=self.part_arr[y_loc,x_loc];
                
        for xval in range(masksize_x):
            for yval in range(masksize_y):
                x_mask=xval-(masksize_x-1)/2;
                y_mask=yval-(masksize_y-1)/2;
                rotated_x=x_mask*np.cos(ran_theta)+y_mask*np.sin(ran_theta);
                rotated_y=x_mask*np.sin(ran_theta)-y_mask*np.cos(ran_theta);
                
                inten=self._gauss_2d(particle_intensity,rotated_x,rotated_y,self.sigma_x,self.sigma_y);
                col=int(x_loc+x_mask);
                row=int(y_loc+y_mask);
                #print (row,col)
                self.part_arr[row,col]=inten;
        return;
    



                 
    def _normalize_image(self):
        """
        Normalizes the image so that it has a mean of zero, and standard
        deviation 1
        """
        if self.scaled_noisy_imgarr!=None:
            return;
        
        if self.noisy_imgarr == None:
            raise AttributeError("Noisy image array not defined")
        
        self.scaled_noisy_imgarr=copy.deepcopy(self.noisy_imgarr);
        mean_val=np.mean(self.noisy_imgarr);
        stddev_val=np.std(self.noisy_imgarr);
        
        for row in range(self.height):
            for col in range(self.width):
                self.scaled_noisy_imgarr[row][col]=float((self.noisy_imgarr[row][col]-mean_val)/stddev_val);
        return;
    
    def _normalize_lowres(self):
        if self.scaled_noisy_lowres!=None:
            return;
        
        if self.noisy_lowres == None:
            raise AttributeError("Noisy Low Resolution image array not defined")
        
        self.scaled_noisy_lowres=copy.deepcopy(self.noisy_lowres);
        mean_val=np.mean(self.noisy_lowres);
        stddev_val=np.std(self.noisy_lowres);
        
        for row in range(self.lowres_height):
            for col in range(self.lowres_width):
                self.scaled_noisy_lowres[row][col]=float((self.noisy_lowres[row][col]-mean_val)/stddev_val);
        return;
        

    def _add_shot_noise(self,img_mat):
        """
        Adds Poisson noise
        """
        (height,width)=img_mat.shape;
        for row in range(height):
            for col in range(width):
                img_mat[row][col]=np.random.poisson(img_mat[row][col]);
        return;
        
    
                
        
    
