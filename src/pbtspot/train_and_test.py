"""
This is the most top-level class which 
(1)gets training examples
(2) initializes and trains a boosting tree
(3) gets test examples
(4) tests the boosting tree

All system level constants are set in this class
"""

#global module imports
from __future__ import division
import os
import pickle
import logging 

#local module imports
import boosting_tree,tree_node
from adaboost import example_images,example_lowres
import confusion_matrix
import globalconstants
#todo: true positives, false positives
logging.basicConfig(filename='logfile.txt',level=logging.DEBUG,filemode='w')


class TrainAndTest:
    """
    trains a boosting tree and tests it
    This is a top level function.
    So this should be very readable to 
    anyone using the program
    public methods:
    def __init__(self,SNR)
    def train_tree(self)
    def test_tree(self)
    def get_roc_curve(self)
    def get_precision_recall_curve(self)
    """
    
    def __init__(self,SNR):
        #Some high level constants defined
        
        #To be defined later
        self.training_examples=None;
        self.testing_examples=None;
        self.boosting_tree=None;
        self.training_accuracy=None;
        self.testing_accuracy=None;
        self.training_eg_img=None;
        self.testing_eg_img=None;

        #define confusion matrix
        self.conf_mat_list=None;
        self._current_conf_matrix=None;
        self._precision=None;
        self._recall=None;
        self._tpr=None;#true positive rate
        self._fpr=None;#false positive rate
        
        
        self.SNR=SNR;
        
        self.eg_imgs=None;
        self.set_constants();
        
    
    def set_constants(self):
        """
        important function to set global constants
        for spot detection
        """
        #transfer all these to globalconstants
        glob_const=globalconstants.GlobalConstants();
        self.maxdepth=glob_const.maxdepth_tree;
        self.no_pos_training_eg=glob_const.no_pos_training_eg;
        self.no_neg_training_eg=glob_const.no_neg_training_eg;
        self.no_pos_testing_eg=glob_const.no_pos_testing_eg;
        self.no_neg_testing_eg=glob_const.no_neg_testing_eg;
        self.prob_threshold=glob_const.prob_threshold; #prob threshold below which object is not detected
        self.prob_threshold_list=[0.01*x for x in range(0,105,5)];
    
        self.test_img_width=512;
        self.test_img_height=512;

        self._is_lowres=glob_const.is_lowres;
        self.test_img_particle_rows=None;
        self.test_img_particle_cols=None;
        self._savetreefile='data/_boostingtree.pickle'


        
        
        #have to put in overfitting constant somewhere
        return;
        
    def train_tree(self,ext_training_eg=None):
        """
        This is the only function that needs to be called externally to train a boosting tree
        (a)fetches training examples
        (b)trains a tree
        (c) gets training accuracy
        
        @defines: self.boosting Tree: Class BoostingTree
        """
        #delete the feature matrix file
        #_response_matrix_filename='data/response_matrix.pickle';
        #if os.path.isfile(_response_matrix_filename) ==True:
        #    os.remove(_response_matrix_filename)
            
        if ext_training_eg==None:
            self._get_examples('Training');
        else:
            self.training_examples=ext_training_eg;#deepcopy?
        self.boosting_tree=boosting_tree.BoostingTree(self.training_examples,\
                                                      self.maxdepth);
                                                      
        self.boosting_tree.train_tree();
        self.get_training_accuracy();

        logging.debug("Training Accuracy:")
        logging.debug(str(self.training_accuracy))
        #delete the feature matrix file
       
        #if os.path.isfile(_response_matrix_filename) ==True:
        #    os.remove(_response_matrix_filename)
        
        
        return;    
    
    def save_tree(self):
        if self.boosting_tree == None:
            raise AttributeError("Boosting Tree not defined yet")
        filename=self._savetreefile;
        pickle.dump(self.boosting_tree,file(filename,'w'));
        return;
        
    def load_saved_tree(self):
        filename=self._savetreefile;
        if os.path.isfile(filename) ==True:
            self.boosting_tree=pickle.load(file(filename));
        else:
            raise AttributeError("Saved Tree does not exist")
        
    def test_tree(self):
        """
        generates test examples and tests tree
        @defines: self.testing accuracy: float between 
        0. and 1.
        
        """
        self._get_examples('Testing')
        tot_test_eg=len(self.testing_examples);
        self.testing_accuracy=0.;
        self._init_confusion_matrix();
        for example in self.testing_examples:
            prob_val=self._test_one_example(example);
            self._calc_confusion_matrix(example,prob_val)
            #if self._detect_test_example(prob_val,example)==True:
            #    self.testing_accuracy+=1;
        
        #self.testing_accuracy=self.testing_accuracy/tot_test_eg;
        
        #modified calculation of test accuracy
        
        (self.testing_accuracy,self.prob_threshold)=self._calc_testing_accuracy();
        self._print_loginfo();#logging
        return;

    def post_prune_tree(self):
        #pass
        #load the testing examples
        #test the original tree if not already not done
        if self.testing_accuracy==None:
            self.test_tree();
                
        self._prune_tree_node(self.boosting_tree.root);
        return
    
    
    def _prune_tree_node(self,treenode):
        
        if treenode.isaleaf==True:
            return
        
        #for now, set isaleaf to true
        treenode.isaleaf=True;
        
        #test the tree again
        testing_acc,prob_threshold=self._calc_testing_accuracy();
        
        #if the accuracy improved, set it to leaf and return
        #update test accuracy so that other instances of function
        #compares with updated testing_accuracy
        
        acc_improve_epsilon=0.05;#how much improvement
        
        if testing_acc-self.testing_accuracy>acc_improve_epsilon:
            self.testing_accuracy=testing_acc;
            self.prob_threshold=prob_threshold;
            return;
        else:
            treenode.isaleaf=False;#restore
            self._prune_tree_node(treenode.leftnode);
            self._prune_tree_node(treenode.rightnode);
            return;
    
            
        

    #not sure this works any more. Fix this.
    def test_tree_whole_image(self):
        """
        generates a test image and tests tree on all windows of the image
        @defines: self.testing accuracy: float between 
        0. and 1.
        """
        
        if self.eg_imgs == None:
            self.eg_imgs=example_images.ExampleImages(self.SNR);
 
        self.testing_accuracy=0.;#use a different variable for this test?
        examples_tested=0;
        self._init_confusion_matrix();
        for newexample in self.eg_imgs.test_fullimage_generator(self.test_img_width,\
                                                            self.test_img_height,\
                                                            self.SNR,\
                                                            self.test_img_particle_rows,\
                                                            self.test_img_particle_cols):
                        

            prob_val=self._test_one_example(newexample);
            examples_tested+=1;

            self.calc_confusion_matrix(newexample);
            
            if self.detect_test_example(prob_val,newexample)==True:
                    self.testing_accuracy+=1;
        
        self.testing_accuracy=self.testing_accuracy/examples_tested;
        return;                
             
    
    def _calc_testing_accuracy(self):
        testing_acc_list=[];
        for index in range(len(self.prob_threshold_list)):
            testing_acc_list.append(self.conf_mat_list[index].get_testing_accuracy());
        
        testing_accuracy=max(testing_acc_list);
        logging.debug("Testing_acc_list:")
        logging.debug(str(testing_acc_list))
        maxindex=testing_acc_list.index(max(testing_acc_list));
        prob_threshold=self.prob_threshold_list[maxindex];
        self._current_conf_matrix=self.conf_mat_list[maxindex];
        return (testing_accuracy,prob_threshold)
        

        
    #make a separate class for fetching train and test examples    
    def _get_examples(self,exampletype):
        #get training examples
        
        #load saved examples or load new examples?
        if exampletype=='Training':
            if self.training_eg_img == None:
                self.training_eg_img=self._assign_example_image(self.no_pos_training_eg,\
                                                                self.no_neg_training_eg);
                self.training_examples=self.training_eg_img.load_examples('Training');
                
        elif exampletype =='Testing':
            if self.testing_eg_img == None:
                self.testing_eg_img=self._assign_example_image(self.no_pos_testing_eg,\
                                                               self.no_neg_testing_eg)
                self.testing_examples=self.testing_eg_img.load_examples('Testing');
                #this is important, ensures that we use the same testing examples everytime
                self.testing_eg_img.save_examples('Testing');
            
        else:
            raise ValueError("ExampleType should be Training or Testing")
        
        return;
    
    
    def _assign_example_image(self,pos_eg,neg_eg):
        if self._is_lowres== False:
            eg_img=example_images.ExampleImages(self.SNR,\
                                                pos_eg,
                                                neg_eg)
        elif self._is_lowres == True:
             eg_img=example_lowres.LowresImages(self.SNR,\
                                                pos_eg,
                                                neg_eg)
        else:
            raise ValueError("_is_lowres should be True or False");
        return eg_img;


        
    
    def get_training_accuracy(self):
        """
        runs the detection over the training examples, and find the 
        accuracy
        """
        tot_training_eg=len(self.training_examples);
        self.training_accuracy=0.;
        
        for example in self.training_examples:
            prob_val=self._test_one_example(example);
            if self._detect_test_example(prob_val,example)==True:
                self.training_accuracy+=1;
        
        self.training_accuracy=self.training_accuracy/tot_training_eg;
        return;
    
    def _test_one_example(self,example):
        """
        given a single example, returns the posterior 
        probability, if tree is already trained
        @param: example: Class Examples
        @retval: probalbility value, float between 0. and 1.
        """
        if self.boosting_tree==None:
            raise ValueError("Boosting Tree not defined")
        
        return self.boosting_tree.compute_posterior(example);
    

    
    def _detect_test_example(self,prob_val,example):
        """
        detects whether the result matches the example label
        @retval: Boolean. True, if result matched label, 
        false otherwise
        """
        detected_value=None;
        
        if prob_val>self.prob_threshold:
            detected_value=True;
        else:
            detected_value=False;
        
        if detected_value==example.label:
            #print "label="+str(example.label) +","+"value="+str(prob_val)
            return True
        else:
            return False;
    
            
    def _calc_confusion_matrix(self,example,prob_val):
    	"""
    	calculates, for each value of probability threshold
    	the values in the confusion matrix
    	"""
    	
        if prob_val>self.prob_threshold:
            self._current_conf_matrix.add_to_matrix(example,True)
        else:
            self._current_conf_matrix.add_to_matrix(example,False)
        
        
        for index in range(len(self.prob_threshold_list)):
            threshold=self.prob_threshold_list[index];
            if prob_val>threshold:
                self.conf_mat_list[index].add_to_matrix(example,True)
            else:
                self.conf_mat_list[index].add_to_matrix(example,False)
                
        return;

    def _init_confusion_matrix(self):
    	"""
    	initializes all variables in the confusion matrix to a list of zeros.
    	given prob_threshold_list is defined
    	"""
        
        self._current_conf_matrix=confusion_matrix.ConfusionMatrix();
    	if self.prob_threshold_list is None:
    		raise ValueError('Probability threshold list is not defined,cannot initialize matrix');
        self.conf_mat_list=[];
    	listlength=len(self.prob_threshold_list);
        for i in range(listlength):
            self.conf_mat_list.append(confusion_matrix.ConfusionMatrix());
        return;
    	

    def get_precision_recall_curve(self):
    	"""
		given the confusion matrix, calculates precision and recall
		for each probability threshold
    	"""
        self._precision=[];
        self._recall=[];
        for matrix in self.conf_mat_list:
            [precision,recall]=matrix.get_precision_recall()
            self._precision.append(precision);
            self._recall.append(recall)
    	return[self._precision,self._recall];
    
    def get_roc_curve(self):
        """
        
        
        """
        self._tpr=[];
        self._fpr=[];
        
        for matrix in self.conf_mat_list:
            [tpr,fpr]=matrix.get_tpr_fpr()
            self._tpr.append(tpr);
            self._fpr.append(fpr)
    	return[self._tpr,self._fpr];
            
    
    def _print_loginfo(self):
        logging.debug(self._current_conf_matrix.print_confusion_matrix());
        
    				 
    		
    		



    	    
                                                    
