"""
@package: test_currimage
@author:Sourav Chatterjee
@brief: test example_images.py
"""
#import from std modules
from __future__ import division
import nose.tools
import numpy as np
import os
#import from local modules
from src.pbtspot.adaboost import example_images
reload(example_images)

class TestExampleImages:
    
    def setup(self):
        self.SNR=2;
        self.num_pos_eg=2;
        self.num_neg_eg=3;
        self.eg_img=example_images.ExampleImages(self.SNR,self.num_pos_eg,self.num_neg_eg);
        
    
    def test_gen_pos_examples(self):
        
        for neweg in self.eg_img._gen_pos_examples(): 
            example= neweg;
            break;
        
        #self.eg_img.show_example('Positive');
        nose.tools.assert_equal(example.label,True);
        nose.tools.assert_equal(example.weight,float(0.5/self.num_pos_eg));
       
            
    def test_gen_neg_examples(self):
        for neweg in self.eg_img._gen_neg_examples(): 
            example= neweg;
            break;
        #self.eg_img.show_example('Positive');
        nose.tools.assert_equal(example.label,False);
        nose.tools.assert_equal(example.weight,float(0.5/self.num_neg_eg));
        
        
    def test__get_new_examples(self):
        self.eg_img.training_examples=[];
        
        self.eg_img._get_new_examples('Positive');
        
        for i in range(self.num_pos_eg):
            nose.tools.assert_equal(self.eg_img.training_examples[i].label,True);
        self.eg_img.training_examples=[];
        self.eg_img._get_new_examples('Negative');
        for i in range(self.num_pos_eg):
            nose.tools.assert_equal(self.eg_img.training_examples[i].label,False);
        self.eg_img.training_examples=None;
        return;
    
    def test_get_new_examples(self):
        training_examples=self.eg_img.get_new_examples();
        nose.tools.assert_equal(len(training_examples),self.num_pos_eg+self.num_neg_eg);
        return;
    
    def test_save_examples(self):
        training_examples=self.eg_img.get_new_examples();
        self.eg_img.save_examples('Training');
        filename="data/"+"Training"+'_examples.pickle';
        fileexists=os.path.isfile(filename);
        nose.tools.assert_equal(fileexists,True)
        return;
    
    def test_load_examples(self):
        filename="data/"+"Training"+'_examples.pickle';
        if os.path.isfile(filename):
            os.remove(filename)
        print "Must load new examples:"
        training_examples=self.eg_img.load_examples('Training');
        self.eg_img.save_examples('Training');
        print "Must load saved examples:"
    
        training_examples=self.eg_img.load_examples('Training');
        self.eg_img._num_neg_eg=self.num_neg_eg+1;
        print "must show not enough saved examples:"
        self.eg_img.load_examples('Training')
                        

    
        
    
    
        
            