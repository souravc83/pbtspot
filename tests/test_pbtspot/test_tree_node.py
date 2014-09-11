"""
@author: Sourav Chatterjee
@brief: tests tree_node.py
"""

#import from std modules
import nose.tools
import logging
import copy

#import from local modules
from src.pbtspot import tree_node
reload(tree_node)
from src.pbtspot.adaboost import example_images, strong_detector
reload(example_images)
reload(strong_detector)


class TestTreeNode:
    def setup(self):
        #keep SNR high
        self.SNR=10;
        self.eg_img=example_images.ExampleImages(self.SNR,2,2);
        training_examples=self.eg_img.load_examples('Training');
        self.eg_img.save_examples('Training')
        
        #default treenode
        self.treenode=tree_node.TreeNode(training_examples,0)
    
  
    def test_check_weights(self):
        eg_img1=example_images.ExampleImages(self.SNR,2,0)
        training_eg=eg_img1.load_examples('Training');
        training_eg[0].weight=0.5;
        training_eg[1].weight=0.5;
        
        
        treenode1=tree_node.TreeNode(training_eg,1);
        nose.tools.assert_equal(treenode1.check_weights(),True);
        
        treenode1=tree_node.TreeNode(training_eg,1);
        treenode1.training_set[1].weight=1;
        nose.tools.assert_equal(treenode1.check_weights(),False);
        return;
    
    def test_train_classifier(self):
        #with the high SNR value set, it is reasonable to assume
        #adaboost will label the examples right
        #this test depends on that assumption
        training_eg=self.eg_img.load_examples('Training');
        treenode1=tree_node.TreeNode(training_eg,depth=0,maxdepth=1);
        treenode1.train_classifier();
        
        #check if correct
        detector=strong_detector.Detector(treenode1.adaboost_rule);
        
        for example in treenode1.training_set:
            nose.tools.assert_equal(example.label,detector.get_binary_decision(example));
        return;
            
        
    def test_calc_adaboost_prob(self):
        
        nose.tools.assert_equal(self.treenode.calc_adaboost_prob(0),0.5);
        nose.tools.assert_almost_equal(self.treenode.calc_adaboost_prob(-1),0.1192,places=4);
        nose.tools.assert_almost_equal(self.treenode.calc_adaboost_prob(1),0.8808,places=4);
        return
    
    def test_assign_child_training_set(self):
         treenode1=tree_node.TreeNode([],1);
         eg_img1=example_images.ExampleImages(self.SNR,1,0)
         training_examples=eg_img1.load_examples('Training');
         example=training_examples[0]
         treenode1.assign_child_training_set(example,.9);
         nose.tools.assert_equal(treenode1.right_training_set,[example]);
         nose.tools.assert_equal(treenode1.left_training_set,None);
         
         treenode1.right_training_set=None;
         treenode1.assign_child_training_set(example,.1);
         nose.tools.assert_equal(treenode1.left_training_set,[example]);
         nose.tools.assert_equal(treenode1.right_training_set,None); 
         
         #When close to 0.5, does example go to both child sets?
         #in right node, example is a deepcopy, so not equal
         treenode1.left_training_set=None;
         treenode1.assign_child_training_set(example,.51);
         nose.tools.assert_equal(treenode1.left_training_set,[example]);
         nose.tools.assert_not_equal(treenode1.right_training_set,[example]);
         nose.tools.assert_equal(len(treenode1.right_training_set),1);
         return;
    
    def test_normalize_weight(self):
        treenode1=tree_node.TreeNode([],1);
        eg_img1=example_images.ExampleImages(self.SNR,2,0)
        pos_examples=eg_img1.load_examples('Training');
        pos_examples[0].weight=1.;
        pos_examples[1].weight=1.;
        treenode1.left_training_set=pos_examples;
        treenode1.normalize_weight('left');
        for example in treenode1.left_training_set:
            nose.tools.assert_almost_equal(example.weight,0.50,places=2);
        treenode1.right_training_set=pos_examples;
        pos_examples[0].weight=1.;
        pos_examples[1].weight=1.;
        treenode1.normalize_weight('right');
        for example in treenode1.right_training_set:
            nose.tools.assert_almost_equal(example.weight,0.50,places=2);
        
        #nose.tools.raises(ValueError,treenode1.normalize_weight,'other');
        return;
    
    def test_calc_empirical_dist(self):
        eg_img1=example_images.ExampleImages(self.SNR,1,1)
        training_examples=eg_img1.load_examples('Training');
        treenode1=tree_node.TreeNode([],1);
        treenode1.training_set=training_examples;
        treenode1.calc_empirical_dist();
        
        for i in range(2):
            nose.tools.assert_almost_equal(treenode1.empirical_dist[i],0.5);
        return;
    
    def test_create_children(self):
        eg_img1=example_images.ExampleImages(self.SNR,2,0)
        training_eg=eg_img1.load_examples('Training')
        pos_training_eg=training_eg[0:2];
        treenode1=tree_node.TreeNode(pos_training_eg,depth=0,maxdepth=2);
        treenode1.create_children();
        #should be perfectly classified, since only positive examples are given
        nose.tools.assert_equal(treenode1.isaleaf,True);
        
        #both positive and negative examples
        eg_img2=example_images.ExampleImages(self.SNR,2,2)
        training_eg=eg_img2.load_examples('Training')
        treenode2=tree_node.TreeNode(training_eg,depth=0,maxdepth=2);
        treenode2.create_children();
        left_examples=treenode2.leftnode.training_set;
        right_examples=treenode2.rightnode.training_set;
        
        for example in left_examples:
            nose.tools.assert_equal(example.label,False);
        for example in right_examples:
            nose.tools.assert_equal(example.label,True);
        
        #we had given 2 postive and 2 negative examples
        nose.tools.assert_equal(len(left_examples),2);
        nose.tools.assert_equal(len(right_examples),2);
        #since maxdepth is 2, the left and righnode must be leaves
        nose.tools.assert_equal(treenode2.isaleaf,False);
        nose.tools.assert_equal(treenode2.leftnode.isaleaf,True);
        nose.tools.assert_equal(treenode2.rightnode.isaleaf,True);
        
        return;
        
    def test_compute_node_posterior(self):
        #train a tree of depth 2, and see if test examples (positive and 
        #negative) give reasonable values of posterior probability
        training_eg=training_eg=self.eg_img.load_examples('Training');
        treenode1=tree_node.TreeNode(training_eg,depth=0,maxdepth=2);
        treenode1.create_children();
        
        test_examples=self.eg_img.load_examples('Testing');
        
        test_pos_examples=test_examples[0:2];
        test_neg_examples=test_examples[2:4];
        
        #the positives should be very close to 1.,negatives should be 
        #very close to 0.
        #I would be concerned if they are not, given the high SNR value
        #keep the print statements to make sure they are.
        
        print "Posterior values for positive examples:"
        
        for example in test_pos_examples:
            prob_val=treenode1.compute_node_posterior(example);
            print prob_val
            nose.tools.assert_true(prob_val>0.5);
        
        print "Posterior values for negative examples:"
        
        for example in test_neg_examples:
            prob_val=treenode1.compute_node_posterior(example);
            print prob_val
            nose.tools.assert_true(prob_val<0.5);
        
        return;
        