#import from std modules
from __future__ import division

#import from local modules
import tree_node

class BoostingTree:
    """
    This class initiates and trains a boosting tree.
    It also calculates the posterior probability p(y|x)
    for a given example x
    """
    
    def __init__(self,training_examples,maxdepth=5):
        self.maxdepth=maxdepth;
        
        #to be defined later
        self.root=None;
        self.training_examples=training_examples;
        self.root=None; #this is the root of the tree
    
    def train_tree(self):
        
        #first normalize weight of training examples before you feed them to root
        self.normalize_weight();
        
        self.root=tree_node.TreeNode(self.training_examples,0,self.maxdepth);
        #Create children recursively
        self.root.create_children();
        return;
    
    def compute_posterior(self,example):
        """
        this is the function called for testing.
        The only function that needs to be called to 
        test an example
        Important: any implementation should not modify boosting tree
        """
        #Important: this should not change the tree in any way
        if self.root ==None:
            raise ValueError("Root not defined yet");
        #error if tree is not trained??
        
        return self.root.compute_node_posterior(example);
    
    def normalize_weight(self):
        no_pos_eg=0;
        no_neg_eg=0;
        for example in self.training_examples:
            if example.label==True:
                no_pos_eg+=1;
            elif example.label==False:
                no_neg_eg+=1;
            else:
                raise ValueError("Example label must be positive or negative");
        
        for example in self.training_examples:
            if example.label==True:
                example.weight=float(0.5/no_pos_eg);
            else:
                example.weight=float(0.5/no_neg_eg);
        return;
    
        
        
        
            
        
        
        