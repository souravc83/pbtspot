#std module imports
from __future__ import division
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import logging

#local module imports
from adaboost import trainer
reload(trainer)

from adaboost import strong_detector
reload(strong_detector)

import globalconstants
logging.basicConfig(filename='logfile.txt',level=logging.DEBUG,filemode='w')

class TreeNode:
    """
    represents a single node in the probabilistic boosting tree
    public methods:
    def __init__(self,training_set,depth,maxdepth=5)
    def create_children(self)
    def compute_node_posterior(self,example)    
    """
    
    def __init__(self,training_set,depth,maxdepth=5):
        
        glob_const=globalconstants.GlobalConstants();
        self.depth=depth;
        self.training_rounds=glob_const.training_rounds;
        self.overfitting_e=glob_const.overfitting_e;#to control overfitting, epsilon in Tu paper
        self.conv_theta=0.45;#to quit if Adaboost convergence is slow
        self._gamma=glob_const.gamma;
        self.maxdepth=maxdepth;
        #to be defined later
        self.training_set=training_set;
        #check that training examples which arrive have proper weights
        if self.check_weights()==False:
            raise ValueError("Example weights must add up to 1.")
        self.nodetrainer=None;
        self.adaboost_rule=None;
        
        self.leftnode=None;
        self.rightnode=None;
        self.left_training_set=None;#training set to be assigned to left tree
        self.right_training_set=None;#training set to be assigned to right tree
        self.empirical_dist=None;#defined as [(y=False) (y=True)]
        self.strong_detector=None;
        self.isaleaf=None; #boolean, whether node is leaf or not
        self._qposlist=None;
            
    def check_weights(self):
        if self.training_set==None:
            raise ValueError("Training set not defined yet");
        if len(self.training_set)==0:
            return True
            #raise ValueError("Cannot define node with empty training set")
        tot_weight=0.;
        
        for example in self.training_set:
            tot_weight+=example.weight;
                        
        if math.fabs(tot_weight-1.0)<0.001: #implement for floating point error
            return True
        else:
            print "Failed Weight Check:"+str(tot_weight)
            return False
    
    def train_classifier(self):
        #set no. of committees, set theta to break early if convergence is slow
        self.nodetrainer=trainer.Trainer(self.training_rounds,self.conv_theta);
        
        #give training examples
        self.nodetrainer.training_examples=copy.deepcopy(self.training_set);

        #set training set
        self.adaboost_rule=self.nodetrainer.run_adaboost();
        return;
       
    
    def calc_adaboost_prob(self,decision):
        """
        finds q(+1|x) in Tu paper
        
        """
        q_positive=1./(1.+np.exp(-2*self._gamma*decision));

        return q_positive;
    
    def assign_child_training_set(self,example,q_pos):
        """
        assigns example to left child, right child or both
        depending on value of q(+1|x) and q(-1|x)
        @param: example: instance of class Example
        @param: q_pos: float between 0. and 1., giving q(+1|x)
        """
        q_neg=1.0-q_pos; #this should be okay, right? we don't need to calculate 
        #q_neg separately;
        
        if (q_pos-0.5)>self.overfitting_e:
            if self.right_training_set is None:
                self.right_training_set=[];
            self.right_training_set.append(example);
        
        elif (q_neg-0.5)>self.overfitting_e:
            if self.left_training_set is None:
                self.left_training_set=[];
            self.left_training_set.append(example);
        
        else:
            if self.right_training_set is None:
                self.right_training_set=[];
            if self.left_training_set is None:
                self.left_training_set=[];
            self.left_training_set.append(example);
            #this is very important
            #so that two nodes don't change the weights of same example seperately
            example_copy=copy.deepcopy(example);
            self.right_training_set.append(example_copy);
        return;
        
    def normalize_weight(self,leftorright):
        
        if leftorright=='left':
            this_training_set=self.left_training_set;
        elif leftorright=='right':
            this_training_set=self.right_training_set;
        else:
            raise ValueError('input either left or right')
        
        if this_training_set==None:
            raise ValueError('Training set is empty')
        
        tot_weight=0.;
        
        for example in this_training_set:
            if example.weight>1 or example.weight<0:
                raise ValueError("Example weight must be between 0 and 1")
            tot_weight+=example.weight;
        for example in this_training_set:
            example.weight=example.weight/tot_weight;
        return;
    
    def calc_empirical_dist(self):
        """
        calculates empirical distribution
        """ 
        if self.training_set is None:
            raise ValueError("Training Set not defined")
        
        self.empirical_dist=[0.,0.]    
        for example in self.training_set:
            if example.label==False:
                self.empirical_dist[0]=self.empirical_dist[0]+example.weight;
            else:
                self.empirical_dist[1]=self.empirical_dist[1]+example.weight;
        
        return;
        
        
                            
    def create_children(self):
        """
        This is the only function that is called by the main boosting tree
        in order to train the tree
        
        """
        if self.training_set is None:
            raise ValueError("Training Set not defined")
        
        #calculate empirical distribution
        self.calc_empirical_dist();
        
        
        #if we've reached maximum depth, make this a leaf
        if self.depth ==(self.maxdepth-1):
            self._print_loginfo();
            self.isaleaf=True;
            return
            
            
        #train the classifier
        self.train_classifier();
        
        #initialize detector
        self.strong_detector=strong_detector.Detector(self.adaboost_rule);#define this
        
        #debug
        #print "Node starts....."  
        self._qposlist=[];      
                    
        for example in self.training_set:
            decision=self.strong_detector.get_decision_stump(example);
            q_pos=self.calc_adaboost_prob(decision);
            self.assign_child_training_set(example,q_pos);
            self._qposlist.append([q_pos,example.label]);#debug
            #print q_pos#debug
        
        #self._print_histogram();
        self._print_loginfo();
        #print "Node Ends......."#debug
        
        #if we've reached maximum depth, make this a leaf
        #if self.depth ==(self.maxdepth-1):
            #self._print_loginfo();
         #   self.isaleaf=True;
          #  return
        
        #if perfectly classified, make this aleaf
        if self.left_training_set ==None or \
           self.right_training_set == None:
            self.isaleaf=True;
            #print "Perfectly Classified"#debug
            return;
            
        #not a leaf if we've reached here
        self.isaleaf=False;
        
        #normalize weights for examples in left and right decision trees    
        self.normalize_weight('left');
        self.normalize_weight('right');
        
        #initialize children
        self.leftnode=TreeNode(self.left_training_set,self.depth+1,self.maxdepth);
        self.leftnode.create_children();
        
        self.rightnode=TreeNode(self.right_training_set,self.depth+1,self.maxdepth);
        self.rightnode.create_children();
        
        return;
    
    def compute_node_posterior(self,example):
        """
        this is the only function that needs to be called to test the examples
        Important: This should not overwrite any tree variables.
        
        """
        
        
        #either both left and right nodes exists or it is a leaf
        
        #if it is leaf
        if self.isaleaf== None:
            raise ValueError("self.isaleaf not defined. run self.create_children() first");
        if self.isaleaf==True:
            posterior_prob=self.empirical_dist[1];
            return posterior_prob;
        
        decision_val=self.strong_detector.get_decision_stump(example);
        q_pos=self.calc_adaboost_prob(decision_val)
        q_neg=1.-q_pos;
        
        #Code when left and right trees both exist        
        if (q_pos-0.5)>self.overfitting_e:
            p_right=self.rightnode.compute_node_posterior(example);
            p_left=self.leftnode.empirical_dist[1];
        
        elif (q_neg-0.5)>self.overfitting_e:
            p_right=self.rightnode.empirical_dist[1];
            p_left=self.leftnode.compute_node_posterior(example);
        
        else:
            p_right=self.rightnode.compute_node_posterior(example);
            p_left=self.leftnode.compute_node_posterior(example);
            
        posterior_prob=q_pos*p_right+q_neg*p_left;
        
        #print [decision_val,q_pos,p_right,q_neg,p_left]
        
        return posterior_prob;
    
    def _print_histogram(self):
        if self._qposlist is None:
            raise AttributeError("qposlist not defined")
            
        _positivelist=[];
        _negativelist=[];
        
        for val in self._qposlist:
            if val[1]==True:
                _positivelist.append(val[0])
            else:
                _negativelist.append(val[0])
        
                
            
        plt.figure();
        plt.title(str(self.depth));
        plt.hist(np.array(_positivelist),bins=10,range=(0,1),color='r',label='Positive',alpha=0.75)
        
        plt.hist(np.array(_negativelist),bins=10,range=(0,1),color='b',label='Negative',alpha=0.75)
        #plt.axis('off')
        plt.ylim(0,250);
        #plt.box('on')
        #plt.xlabel("Posterior Probability")
        #plt.ylabel("Number of Examples")
        
        plt.show();
        
        return;
    
    def _print_loginfo(self):
        logging.debug("Node: ")
        logging.debug("Tree Depth: %d"%(self.depth));
        logging.debug("Empirical distribution: "+str(self.empirical_dist))
        
        
        
        
