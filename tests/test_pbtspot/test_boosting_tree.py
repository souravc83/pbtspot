"""
@author: Sourav Chatterjee
@brief: tests boosting_tree.py
"""

#import from std modules
import nose.tools
import logging
import copy

#import from local modules
from src.pbtspot import boosting_tree,tree_node
reload(boosting_tree)
reload(tree_node)
from src.pbtspot.adaboost import example_images, strong_detector
reload(example_images)
reload(strong_detector)


class TestBoostingTree:
    def setup(self):
        #pass
        #keep SNR high
        self.SNR=10;
        self.num_pos_eg=3;
        self.num_neg_eg=3;
        self.eg_img=example_images.ExampleImages(self.SNR,3,3);
        self.training_eg=self.eg_img.load_examples('Training')

    
    def test_train_tree(self):
        boosting_tree1=boosting_tree.BoostingTree(self.training_eg,2);
        boosting_tree1.train_tree();
        nose.tools.assert_equal(boosting_tree1.root.isaleaf,False);
        nose.tools.assert_equal(boosting_tree1.root.leftnode.isaleaf,True);
        nose.tools.assert_equal(boosting_tree1.root.rightnode.isaleaf,True);
        
        leftnode_examples=boosting_tree1.root.leftnode.training_set;
        rightnode_examples=boosting_tree1.root.rightnode.training_set;
        
        nose.tools.assert_equal(len(leftnode_examples),self.num_neg_eg);
        nose.tools.assert_equal(len(rightnode_examples),self.num_pos_eg);
        return;
    
    def test_compute_posterior(self):
        eg_img1=example_images.ExampleImages(self.SNR,3,3);
        training_eg2=eg_img1.load_examples('Training')
        
        boosting_tree1=boosting_tree.BoostingTree(training_eg2,2);
        boosting_tree1.train_tree();
        eg_img2=example_images.ExampleImages(self.SNR,3,3);
        test_examples=eg_img2.load_examples('Testing')
        test_pos_examples=test_examples[0:3];
        test_neg_examples=test_examples[3:6];
        
        #the positives should be very close to 1.,negatives should be 
        #very close to 0.
        #I would be concerned if they are not, given the high SNR value
        #keep the print statements to make sure they are
        
        print "Posterior values for positive examples:"
        
        for example in test_pos_examples:
            prob_val=boosting_tree1.compute_posterior(example)
            print prob_val
            nose.tools.assert_true(prob_val>0.5);
            
        
        print "Posterior values for negative examples:"
        
        for example in test_neg_examples:
            prob_val=boosting_tree1.compute_posterior(example);
            print prob_val
            nose.tools.assert_true(prob_val<0.5);
        
        return;
    def test_highnoise_example(self):
        SNRval=0.5;
        eg_img1=example_images.ExampleImages(0.5,60,60);
        training_eg2=eg_img1.get_new_examples()
        
        boosting_tree1=boosting_tree.BoostingTree(training_eg2,5);
        #check printed values
        boosting_tree1.train_tree();
        return;
    
        
        
        
        
        
        
