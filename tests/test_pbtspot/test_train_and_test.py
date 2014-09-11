"""
@author: Sourav Chatterjee
@brief: tests boosting_tree.py
"""

#import from std modules
import nose.tools
import logging
import copy

#import from local modules
from src.pbtspot import boosting_tree,train_and_test
reload(boosting_tree)
reload(train_and_test)
from src.pbtspot.adaboost import example_images, strong_detector
reload(example_images)
reload(strong_detector)

class TestTrainTest:
    def setup(self):
        #default train and test class
        self.SNRval=10;
        self.trainandtest=train_and_test.TrainAndTest(self.SNRval);
        self.trainandtest.no_pos_training_eg=10;
        self.trainandtest.no_neg_training_eg=10;
        self.trainandtest.no_pos_testing_eg=10;
        self.trainandtest.no_neg_testing_eg=10;
    
    def test_get_examples(self):
        trainandtest1=train_and_test.TrainAndTest(self.SNRval);
        #fails if not called with Training or Testing
        nose.tools.assert_raises(ValueError,trainandtest1._get_examples,'Other')
        trainandtest1._get_examples('Training');
        pos_eg=trainandtest1.no_pos_training_eg;
        neg_eg=trainandtest1.no_neg_training_eg;
        
        nose.tools.assert_equal(len(trainandtest1.training_examples),pos_eg+neg_eg);
        
        for i in range(pos_eg):
            example=trainandtest1.training_examples[i];
            label=example.label
            nose.tools.assert_equal(label,True);
            classname=example.__class__.__name__;
            nose.tools.assert_equal(classname,'Example')
        
        for i in range(neg_eg):
            example=trainandtest1.training_examples[i+pos_eg]
            label=example.label;
            #print label;
            nose.tools.assert_equal(label,False);
            classname=example.__class__.__name__;
            nose.tools.assert_equal(classname,'Example')
            return;
        
    def test_train_tree(self):
        trainandtest1=train_and_test.TrainAndTest(self.SNRval);
        trainandtest1.maxdepth=2;#this is root+left+right
        trainandtest1.no_pos_training_eg=10;
        trainandtest1.no_neg_training_eg=10;
        trainandtest1.no_pos_testing_eg=10;
        trainandtest1.no_neg_testing_eg=10;
        
        trainandtest1.train_tree();
        isrootleaf=trainandtest1.boosting_tree.root.isaleaf;
        isleftnodeleaf=trainandtest1.boosting_tree.root.leftnode.isaleaf;
        isrightnodeleaf=trainandtest1.boosting_tree.root.rightnode.isaleaf;
        
        nose.tools.assert_equal(isrootleaf,False);
        nose.tools.assert_equal(isleftnodeleaf,True);
        nose.tools.assert_equal(isrightnodeleaf,True);
        
        #with a high SNR, training accuracy should be 1. or close
        #I would be cautious if it is not
        train_acc=trainandtest1.training_accuracy;
        print "Training_Accuracy = "+str(train_acc);
        nose.tools.assert_true(train_acc>0.9);
        return;
        
    def test_test_tree(self):
        trainandtest1=train_and_test.TrainAndTest(self.SNRval);
        trainandtest1.maxdepth=2;#this is root+left+right
        trainandtest1.no_pos_training_eg=10;
        trainandtest1.no_neg_training_eg=10;
        trainandtest1.no_pos_testing_eg=10;
        trainandtest1.no_neg_testing_eg=10;
        
        trainandtest1.train_tree();
        trainandtest1.test_tree();
        
        #with a high SNR, testing accuracy should be 1. or close
        #I would be cautious if it is not
        test_acc=trainandtest1.testing_accuracy;
        print "Testing_Accuracy = "+str(test_acc);
        nose.tools.assert_true(test_acc>0.9);
        return;
        
    def test_test_tree_whole_image(self):
        #self.trainandtest.train_tree();
        #self.trainandtest.test_tree_whole_image();
        
        #test_acc=self.trainandtest.testing_accuracy;
        #print "Whole Image Testing_Accuracy = "+str(test_acc);
        #This takes a long time. Think if we really need this
        pass
        return;
    
    def test_get_roc_curve(self):
        self.trainandtest.train_tree();
        self.trainandtest.test_tree();
        [tpr,fpr]=self.trainandtest.get_roc_curve();
        print tpr
        print fpr
        
    
        