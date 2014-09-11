"""
@author: Sourav Chatterjee
@brief: tests boosting_tree.py
"""

#import from std modules
import nose.tools
import logging
import copy

#import from local modules
from src.pbtspot import adaboost_train_test
reload(adaboost_train_test)

from src.pbtspot.adaboost import example_images, strong_detector
reload(example_images)
reload(strong_detector)

class TestAdaboostTrainTest:
    def setup(self):
        self.training_rounds=7;
        pos_eg=10;
        neg_eg=10;
        self.SNR=0.5;

        self.adaboost_tt=adaboost_train_test.AdaBoostTrainTest(self.SNR,\
                          self.training_rounds, pos_eg,neg_eg,pos_eg,neg_eg);

    def test_train_adaboost(self):
        self.adaboost_tt.train_adaboost();

        adaboost_rule=self.adaboost_tt._adaboost_rule;

        
        #nose.tools.assert_equal(len(adaboost_rule),self.training_rounds);
        nose.tools.assert_true(self.adaboost_tt.training_accuracy>0.5);#reasonable
        print "Training Acc: "+str(self.adaboost_tt.training_accuracy);
    
    def test_test_adaboost(self):
        self.adaboost_tt.train_adaboost();
        self.adaboost_tt.test_adaboost();
        nose.tools.assert_true(self.adaboost_tt.testing_accuracy>0.5);#reasonable
        print "Testing Acc: "+str(self.adaboost_tt.testing_accuracy);





