"""
This class trains and tests a standalone Adaboost
strong detector.
The main purpose is to compare the standalone detector
with the tree
"""
#global module imports
from __future__ import division
import os

#local module imports
from adaboost import trainer,example_images,strong_detector
import globalconstants

class AdaBoostTrainTest(object):
    """
    public methods:
    def __init__(self,SNRval,training_rounds,pos_training_eg=10,\
                 neg_training_eg=10,pos_testing_eg=10,neg_testing_eg=10): 
    train_adaboost()
    test_adaboost()

    public variables:
    training_accuracy
    testing_accuracy




    """
    def __init__(self,SNRval,training_rounds,pos_training_eg=10,\
                 neg_training_eg=10,pos_testing_eg=10,neg_testing_eg=10):
        self._SNR=SNRval;
        self._training_rounds=training_rounds;
        self._trainer=None;
        self._adaboost_rule=None;
        self._pos_training_eg=pos_training_eg;
        self._neg_training_eg=neg_training_eg;
        self._pos_testing_eg=pos_testing_eg;
        self._neg_testing_eg=neg_testing_eg;
        self._training_examples=None;
        self._testing_examples=None;
        self._detector=None;
        self.training_accuracy=None;
        self.testing_accuracy=None;
    
        
    
    def train_adaboost(self):
        #delete the feature matrix file
        _response_matrix_filename='data/response_matrix.pickle';
        if os.path.isfile(_response_matrix_filename) ==True:
            os.remove(_response_matrix_filename)
        self._get_examples('Training');
        self._trainer=trainer.Trainer(self._training_rounds);
        self._trainer.training_examples=self._training_examples;
        self._adaboost_rule=self._trainer.run_adaboost();
        
        tot_eg=len(self._training_examples);
        self.training_accuracy=0.;
        for example in self._training_examples:
            if self._test_adaboost_example(example)==True:
                self.training_accuracy+=float(1./tot_eg);
         #delete the feature matrix file       
        if os.path.isfile(_response_matrix_filename) ==True:
            os.remove(_response_matrix_filename)
        return;
    
    def test_adaboost(self):
        self._get_examples('Testing');
        
        tot_eg=len(self._testing_examples);
        self.testing_accuracy=0.;
        for example in self._testing_examples:
            if self._test_adaboost_example(example)==True:
                self.testing_accuracy+=float(1./tot_eg);
        return;
    
    def _test_adaboost_example(self,example):
        if self._detector ==None:
            self._detector=strong_detector.Detector(self._adaboost_rule);
        if self._detector.get_binary_decision(example) ==example.label:
            return True;
        else:
            return False;        
        
        
    
    def _get_examples(self,egtype):
        if egtype!='Training' and egtype!='Testing':
            raise ValueError("egtype must be training or testing")
            
        if egtype=='Training':    
            eg_imgs=example_images.ExampleImages(self._SNR,self._pos_training_eg,self._neg_training_eg);
            self._training_examples=eg_imgs.get_new_examples();
        else:
            eg_imgs=example_images.ExampleImages(self._SNR,self._pos_testing_eg,self._neg_testing_eg);
            self._testing_examples=eg_imgs.get_new_examples();
        return;
                    
