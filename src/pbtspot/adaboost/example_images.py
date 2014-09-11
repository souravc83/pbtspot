"""
@author: Sourav Chatterjee
@brief: Generates training and testing examples from synthetic images
"""

#std module imports
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import random
import os
import pickle
import copy
import uuid

#local module imports

from generateimage import image_generate
reload(image_generate)
from pbtspot import globalconstants



class Example:
    """
    defines the tuple(xi,yi,wi)
    """
    def __init__(self,image,label=True,weight=1.):
        self.image=image;
        self.label=label;
        self.weight=weight;
        self.hashkey=str(uuid.uuid4()); #for storing hashtable of examples
        self.check_values();
        
        
    def check_values(self):
        if self.weight<0 or self.weight>1:
            raise ValueError("Weight must be between 0 and 1");
        
        if (self.label is not True) and (self.label is not False):
            raise ValueError("Label must be True or False") 
            
        return;
    
        

class ExampleImages:
#todo:implement get_saved_positive_examples()    

    """
    This class is used to generate Training and Testing examples, save and load them.
    Public methods:
    def __init__(self,SNR,pos_eg_no=10,neg_eg_no=10)
    def get_new_examples(self)
    def save_examples(self,egtype)
    def load_examples(self,egtype)
    
    
    """
    def __init__(self,SNR,pos_eg_no=10,neg_eg_no=10):
        
        glob_const=globalconstants.GlobalConstants();
        self.global_window_width=glob_const.global_window_width;
        self.global_window_height=glob_const.global_window_height;
        self.SNR=SNR;#we should be able to change SNR from the highest point in the program
        self._img_gen=image_generate.GenerateImage(self.SNR);
        
        self._img_width=self._img_gen.width;
        self._img_height=self._img_gen.height;
        self.num_particles=self._img_gen.part_rows*self._img_gen.part_columns;
        
        #training examples
        self.training_examples=None;
        if pos_eg_no ==0 and neg_eg_no==0:
            raise ValueError("Cannot create instance with 0 positive and 0 negative examples")
        if pos_eg_no>=0:
            self._num_pos_eg=pos_eg_no;#public variable: num_pos_eg
        else:
            raise ValueError("number of positive examples cannot be negative")
        if neg_eg_no>=0:
            self._num_neg_eg=neg_eg_no;#public variable: num_neg_eg
        else:
            raise ValueError("number of positive examples cannot be negative")
        
        #example weights:
        self._pos_example_weights=None;
        self._neg_example_weights=None;
        
        self._set_example_weights();    
            
        self.num_neg_eg_img=self.num_particles;#this is the no. of negative examples generated per image
        
        self._eg_filename='_examples.pickle';
        
        
        #random number generator
        random.seed();
        
        
        #to be defined later
        self.test_img=None;
    
    def get_new_examples(self):
        """
        called externally to give a list of new examples
        if you want to load saved examples, use load_examples(egtype)
        """
        if self.training_examples != None:
            print "Warning: training examples already exist. Do you still want to generate again?"
        
        self.training_examples=[];
        self._get_new_examples('Positive')
        self._get_new_examples('Negative');
        return self.training_examples;
    
    def _get_new_examples(self,egsign):
        
        numeg=0;
        
        if egsign == 'Positive':            
            numeg=self._num_pos_eg;
        elif egsign == 'Negative':
            numeg=self._num_neg_eg;
        else:
            raise ValueError("egsign must be Positive or Negative")
        counter=0;
        
        if egsign == 'Positive':
            if self._num_pos_eg ==0:
                return;
            for example in self._gen_pos_examples():
                
                if counter == numeg:
                    break;
                self.training_examples.append(example);
                counter+=1;
                
        else:
            if self._num_neg_eg ==0:
                return;
            for example in self._gen_neg_examples():
                if counter == numeg:
                    break;
                self.training_examples.append(example);
                counter+=1 
        return;
    
    def save_examples(self,egtype):
        """
        @param: egtype: String-"Training" or "Testing"
        """
        if self.training_examples == None:
            raise AttributeError("Training Examples not defined yet")
        filename=self._get_filename_exampletype(egtype); 
        pickle.dump(self.training_examples,file(filename,'w'));
        return;
        
    
    def load_examples(self,egtype):
        """
        @param: egtype: String-"Training" or "Testing"
        """
        filename=self._get_filename_exampletype(egtype); 
        if os.path.isfile(filename) ==True:
            try:
                self._load_saved_examples(filename);
                print 'Using Saved Examples'
                return self.training_examples;
            except ValueError:
                print 'Not enough saved Examples. Loading New Examples'
                return self.get_new_examples();
        else:
            print 'Loading New Examples'
            return self.get_new_examples();
    
    
    def _get_filename_exampletype(self,egtype):
        """
        @param: egtype: String-"Training" or "Testing"
        """
        if egtype == "Training":
            filename="data/"+"Training"+self._eg_filename;
        elif egtype == "Testing":
            filename="data/"+"Testing"+self._eg_filename; 
        else:
            raise ValueError("Example type must be training or testing")
        return filename;
        
    
    
        
        
    
    def _gen_pos_examples(self):
        """
        generates all positive examples by generating a new image
        @writes to: self.training_examples: list of instances of class Example
       
        """
        if self._num_pos_eg ==0:
            raise ValueError("Number of positive examples is 0");
        
       #should we take normalized image array?
        self._img_gen.reset_image();
        self._img_gen.generate_img();
        [xleft,xright,ytop,ybottom]=self._get_window_parameters();
        tot_images=self.num_particles;
        counter=0;
        while True:
            if counter == tot_images:
                self._img_gen.reset_image();
                self._img_gen.generate_img();
            for location in self._img_gen.particle_loc:
                [x_loc,y_loc]=location;
                
                if self._check_coord(x_loc-xleft,y_loc-ytop) and\
                   self._check_coord(x_loc+xright,y_loc+ybottom):
                    #pos_img=self._img_gen.noisy_imgarr[y_loc-ytop:y_loc+ybottom,\
                    #                                 x_loc-xleft:x_loc+xright];
                    pos_img=self._img_gen.scaled_noisy_imgarr[y_loc-ytop:y_loc+ybottom,\
                                                     x_loc-xleft:x_loc+xright];
                    thisexample=Example(pos_img,True,self._pos_example_weights);
                    
                    counter+=1
                    yield thisexample;
        return;
        
        
    
        
       
    def _gen_neg_examples(self):
        """
        generates all positive examples by generating a new image
        @writes to: self.training_examples: list of instances of class Example
        """
        if self._num_neg_eg == 0:
            raise ValueError("Number of negative examples is 0");
        
        self._img_gen.reset_image();
        self._img_gen.generate_img();
        [xleft,xright,ytop,ybottom]=self._get_window_parameters();
        
        counter=0;
        while True:
            if counter == self.num_neg_eg_img:
                self._img_gen.reset_image();
                self._img_gen.generate_img();
                
            #get random rows and columns
            rowno=random.randint(ytop,self._img_height-ybottom);
            colno=random.randint(xleft,self._img_width-xright);
            
            #Check if this is particle
            if [colno,rowno] in self._img_gen.particle_loc :
                continue;
            elif self._check_coord(colno-xleft,rowno-ytop)==False or\
                 self._check_coord(colno+xright,rowno+ybottom)==False:
                continue;
                    
            else:
                #neg_img=self._img_gen.noisy_imgarr[rowno-ytop:rowno+ybottom,\
                #                             colno-xleft:colno+xright];
                neg_img=self._img_gen.scaled_noisy_imgarr[rowno-ytop:rowno+ybottom,\
                                             colno-xleft:colno+xright];
                thisexample=Example(neg_img,False,self._neg_example_weights);
                counter+=1;
                yield thisexample;          
        return;
          
    
    def _load_saved_examples(self,filename):
        """
        This implementation relies on the fact that positive examples are loaded first
        and negative examples are loaded last. So it searches for positive examples from 
        left of saved array, and negative examples from right of saved array
        """
        try:
            temp_examples=pickle.load(file(filename));
        except:
            raise ValueError("Cannot load saved pickle file")
        self.training_examples=[];
        assign_success=False;
        pos_eg=0;
        neg_eg=0;
        
        tot_eg=len(temp_examples)
        
        #Positive examples, start from left
        for i in range(tot_eg):
            if pos_eg ==self._num_pos_eg:
                break;
            example=temp_examples[i];
            if example.label == True:
                example.weight=self._pos_example_weights;
                self.training_examples.append(example);
                pos_eg+=1;
            else:
                break;
        
        #neg examples, start from right
        for i in range(tot_eg):
            if neg_eg ==self._num_neg_eg:
                break;
            example=temp_examples[tot_eg-i-1];
            if example.label == False:
                example.weight=self._neg_example_weights;
                self.training_examples.append(example);
                neg_eg+=1;
            else:
                break;
                    
        if pos_eg ==self._num_pos_eg and neg_eg==self._num_neg_eg:
            assign_success=True; 
        
        if assign_success == True:
            return;
        else:
            self.training_examples=[];
            raise ValueError("Not enough saved examples")

    
    #probably does not work anymore
    #fix this
    def test_fullimage_generator(self,testwidth=None,testheight=None,testSNR=None,\
                                particle_rows=None,particle_cols=None):
        """
        defines a complete image, to be used for testing purposes
        This is a generator function.
        @yieldval: newexample: Instance of class Example. 
        label is true if there is particle, otherwise false.
        Weight is 1., just a dummy weight.
        
        @param: testwidth: integer, width of test image
        @param: testheight: integer, height of test image
        @param: testSNR: float, SNR of test image
        
        
        """
        if testwidth==None:
            testheight=self._img_width;
        
        if testheight==None:
            testheight=self._img_height;
        
        if testSNR==None:
            testSNR=self.SNR;
        
        test_img_gen=image_generate.GenerateImage(testSNR,testwidth,testheight);
        
        if particle_rows is not None:
            test_img_gen.part_rows=particle_rows;
        if particle_cols is not None:
            test_img_gen.part_columns=particle_cols;
        
        test_img_gen.generate_img();
        test_img=test_img_gen.noisy_imgarr;
        [xleft,xright,ytop,ybottom]=self._get_window_parameters();
        
        for centerx in range(xleft,testwidth-xright):
            for centery in range(ytop,testheight-ybottom):
                test_imgwindow=test_img[centery-ytop:centery+ybottom,\
                                                centerx-xleft:centerx+xright];
                if([centerx,centery] in test_img_gen.particle_loc):
                    testlabel=True;
                else:
                    testlabel=False;
                
                newexample=Example(test_imgwindow,testlabel,1.);
                yield newexample;
        return;
        
    
        
    
    def _check_coord(self,x_loc,y_loc):
        if self._img_width==None or self._img_height==None:
            raise AttributeError("Cannot check coordinates. Height and weight not defined");
        
        if x_loc>=0 and x_loc<self._img_width and \
           y_loc>=0 and y_loc<self._img_height:
            return True;
        else:
            return False;
        
    #does not work anymore
    #fix this
    def show_example(self,eg_type='Positive'):
        """
        shows an example image
        @param: eg_type: String, 'Positive' or 'Negative'
        """
        if(eg_type!='Positive' and eg_type!='Negative'):
            print 'example type should be Poitive or Negative';
            return;
        
        if(eg_type=='Positive'):
            if len(self.pos_training_examples)==0:
                print 'Postive examples not generated'
                return;
                
            pos_img=self.pos_training_examples[-1].image;
        else:
            if len(self.neg_training_examples)==0:
                print 'Negative examples not generated'
                return;
            pos_img=self.neg_training_examples[-1].image;
            
        fig=plt.figure();
        ax=fig.add_subplot(111);
        ax.imshow(pos_img,cmap=cm.Greys_r,interpolation='nearest');
        numrows,numcols=pos_img.shape;
    
        def format_coord(x,y):
            col=int(x);
            row=int(y);
        
            if col>0 and col<numcols and row>0 and row<numrows:
                z=pos_img[row,col];
                return 'x=%1.1f, y=%1.1f, z=%1.1f'%(x, y, z);
            else:
                return 'x=%1.1f, y=%1.1f'%(x, y);
            
        ax.format_coord=format_coord;
        plt.show()
        return;
    
    def _get_window_parameters(self):
        """
        returns a list:[xleft,xright,ytop,ybottom]
        """
        xleft=int(math.ceil(self.global_window_width/2));
        xright=int(math.floor(self.global_window_width/2))+1;
        ytop=int(math.ceil(self.global_window_height/2));
        ybottom=int(math.floor(self.global_window_height/2))+1;
        return [xleft,xright,ytop,ybottom];
    
    def _set_example_weights(self):
        if self._num_pos_eg!=0 and self._num_neg_eg!=0:
            self._pos_example_weights=float(0.5/self._num_pos_eg);
            self._neg_example_weights=float(0.5/self._num_neg_eg);
        elif self._num_pos_eg ==0:
            self._pos_example_weights=0.;
            self._neg_example_weights=float(1./self._num_neg_eg);
        else:
            self._pos_example_weights=float(1./self._num_pos_eg);
            self._neg_example_weights=0.;
            
            
