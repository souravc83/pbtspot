
#import from standard library
from __future__ import division
import logging
import numpy as np
import copy
import pickle
import os

#import local modules
from features import basefeature,diff_features
import example_images,currimage
reload(example_images)
from pbtspot import globalconstants


logging.basicConfig(filename='logfile.txt',level=logging.DEBUG,filemode='w')

#Todo: use examples class to make things more consistent

class Trainer(object):
    """
    This class trains a strong AdaBoost classifier
    public variables:
        self.training_examples: list of instances of class Example
    public methods:
        def __init__(self,training_rounds=10,conv_theta=0.45)
        def run_adaboost(self) #this runs the training algorithm and returns
        self.adaboost_rule: list of lists of the form[alpha,feature] for 
        each training round 
    """
    
    def __init__(self,training_rounds=10,conv_theta=0.45):
    	self.training_rounds=training_rounds;
        #list of features;
        glob_const=globalconstants.GlobalConstants();
        self.global_window_width=glob_const.global_window_width;
        self.global_window_height=glob_const.global_window_height;
        
        #logging.basicConfig(filename='logfile.txt',level=logging.DEBUG)
        self.conv_theta=conv_theta;
        
        #saving stuff
        self._feature_filename='data/feature_list.pickle'
        self._response_matrix_filename='data/response_matrix.pickle'
        
        
        #to be defined later
        self._features=None;
        self._training_examples=None;
        self._no_pos_training_eg=None;
        self._no_neg_training_eg=None;
        
        self.adaboost_rule=None;
        self.log_loss=None;
        self.training_error=None;
        self._int_image_list=None;
        
    @property
    def training_examples(self):
        return self._training_examples;
    
    @training_examples.setter
    def training_examples(self,value):
        self._training_examples=value;
        self._no_pos_training_eg=0;
        self._no_neg_training_eg=0;
            
        for example in self._training_examples:
            if example.label == True:
                self._no_pos_training_eg+=1;
            else:
                self._no_neg_training_eg+=1;
        return;

        
    def run_adaboost(self):
        """
        This is the main function called by an external program.
        This should be able to run by itself with default values
        """
        
        #initialization
        #self._generate_features();
        self._load_features();
        #print "Features Generated....."
        if self._training_examples==None:
            raise AttributeError("Training Examples not given to Adaboost trainer");
        self._init_adaboost();
        #print "Feature Vectors Initialized...."
        self._adaboost_main_loop();
        return self.adaboost_rule;
    
    def _adaboost_main_loop(self):
        """
        this function updates weights, sets alpha values and finds error
        @defines: self.adaboost_rule: list of lists.
        each element is of the form:[alpha_t bestfeature]
        """
        self.adaboost_rule=[];
        self.log_loss=[];
        self.training_error=[];
        
        for thisround in range(self.training_rounds):
            print "Adaboost Training Round: "+str(thisround);
            
            bestfeature=self._get_best_decision_stump();
            #break if already perfectly classified
            #check why errors are so close to zero, with better unit tests
            #this situation can only happen for the first round
            if(bestfeature.error<=1e-10):#floating point problem, put close to zero
                round_error=1e-5;#this is a made up number. see how to handle this case
                #properly
                alpha_t=0.5*np.log((1.0-round_error)/round_error);
                self.adaboost_rule.append([alpha_t, bestfeature]);
                break;
            #break if convergence is really slow
            if(bestfeature.error>=self.conv_theta):
                break;
            #tackle possible floating point problems while calculating alpha_t
            if(bestfeature.error<=1e-5):
                bestfeature.error=1e-5;
            
            round_error=bestfeature.error;
            print "Round Error:"+' '+str(round_error);
            alpha_t=0.5*np.log((1.0-round_error)/round_error);
            copy_bestfeature=copy.deepcopy(bestfeature);#very important
            self.adaboost_rule.append([alpha_t,copy_bestfeature]);
            self._update_weights(bestfeature);
            self._reset_features();
            self._calc_logloss();
            #print "Log Loss: "+str(self.log_loss[thisround]);
            #print "Training Error: "+str(self.training_error[thisround]);
        
        
        #debug
        #all/most debug statements go here;
        #[alpha,testfeature]=self.adaboost_rule[0];
        #logging.debug(testfeature.response_vector);
        #logging.debug(self.training_labels);
        #logging.debug(self.example_weights);
            
        return;      
    
    def _init_adaboost(self):
        """
        initializes weights, and calculates response vector for features
        """
        #this assumes that features and examples are already calculated
        #should 3e put them in here?
        #should we raise exception if not calculated?
         
        if self._features==None:
            print 'Not generated features.Cannot calculate threshold'
            #should we raise error?
            return

        #calculate integral images for all training examples
        self._int_image_list=[]

        for example in self._training_examples:
            currimg=currimage.CurrImage(example.image);
            currimg.calc_integral_image();
            int_image=currimg.int_image;
            self._int_image_list.append(int_image);

        #debug
        #for example in self._training_examples:
        #	print example.hashkey
 

        #calculate response vector for all features
        self._load_response_vector();
        #initialize example weights
        self._init_example_weights();     
        return;
    
    def _load_response_vector(self):
        if os.path.isfile(self._response_matrix_filename) ==True:
            self._load_saved_response_vector();
            print "Something wrong, we don't have a response vector now"

        else:
            #calculate response vector
            for current_feature in self._features:
                self._calc_response_vector(current_feature);
                current_feature.error=2.0;
                current_feature.threshold=0.;
                current_feature.toggle=1;
                current_feature.margin=0; #Why not do these in the BaseFeatures itself
            #self._save_response_matrix();
        return;
        
    def _load_saved_response_vector(self):
        temp_response_matrix=pickle.load(file(self._response_matrix_filename));
        #Very IMPORTANT: features must be generated in the same order
        
        for index in range(len(self._features)):
            current_feature=self._features[index]
            current_feature.error=2.0;
            current_feature.threshold=0.;
            current_feature.toggle=1;
            current_feature.margin=0; #Why not do these in the BaseFeatures itself
            
            current_feature.response_vector=[];
            temp_response_dict=temp_response_matrix[index]
            for egindex in range(len(self._training_examples)):
                try:
                    example=self._training_examples[egindex]
                    response_val=temp_response_dict[example.hashkey];
                    
                    current_feature.response_vector.append((egindex,response_val))
                except:
                	raise
                    #raise AttributeError("Example not found in saved response matrix.\
                    #Are you using response matrix from previous run?")
                
            current_feature.response_vector.sort(key=lambda tup:tup[1]);
        return;
    
    def _save_response_matrix(self):
        response_matrix=[{}]*len(self._features)
        for index in range(len(self._features)):
            current_feature=self._features[index]
            for i in range(len(current_feature.response_vector)):
                row,response_val=current_feature.response_vector[i];
                eg_key=self._training_examples[row].hashkey;
                response_matrix[index][eg_key]=response_val;
        pickle.dump(response_matrix,file(self._response_matrix_filename,'w'));
        return;
            
                
                  
    def _load_features(self):
        #either generates new features, or loads older ones

        #right now just generates new features
        self._generate_features();
        return;

        #older implementation, loads saved features, if available
        #if os.path.isfile(self._feature_filename) ==True:
        #    self._features=pickle.load(file(self._feature_filename));
        #else:
        #    self._generate_features();
        #return;
        
        
    def _generate_features(self):
        #Don't make any more compact, because I want to know
        #how many of each type of feature is created;
        feature_types=[];
        feature_types.append(diff_features.EdgeFeature_1());
        feature_types.append(diff_features.EdgeFeature_2());
        feature_types.append(diff_features.LineFeature_1());
        feature_types.append(diff_features.LineFeature_2());
        feature_types.append(diff_features.LineFeature_3());
        feature_types.append(diff_features.LineFeature_4());
        feature_types.append(diff_features.CenterFeature());
        
        #Initialize feature list
        self._features=[];
        
        for feature in feature_types:
            self._generate_each_feature(feature);
        
        #pickle.dump(self._features,file(self._feature_filename,'w'));
        return;
        
        
                
    def _generate_each_feature(self,this_feature):
        
        if(self._features==None):
            raise AttributeError("features not defined yet");
            
        self.test_featurecount=0;#for debugging purposes
        width_incr=this_feature.width_incr;
        height_incr=this_feature.height_incr;
        
        for width in range(width_incr,self.global_window_width+width_incr,width_incr):
            for height in range(height_incr,self.global_window_height+height_incr,height_incr):

                for x_val in range(0,self.global_window_width-width+1):
                    for y_val in range(0,self.global_window_height-height+1):
                        
                        new_feature=copy.deepcopy(this_feature);#very important
                        
                        new_feature.generate_feature(x_val,y_val,width,height);
                        self._features.append(new_feature);
                        self.test_featurecount+=1;
           
        #print self.test_featurecount; 
        #logging.debug(self.test_featurecount);
        return;
        

    def _calc_response_vector(self,current_feature):
        
        
        if self._training_examples==None:
            raise AttributeError('Not generated training examples.Cannot calculate response vector')
        if self._int_image_list == None:
            raise AttributeError('Not generated integral image list')
        
        response_numrows=len(self._training_examples);
       
        #Initialize response vector
        current_feature.response_vector=[]#response vector is a variable of a feature
        for row in range(response_numrows):
            #takes in the tuple(exampleno, response_value)
            int_image=copy.deepcopy(self._int_image_list[row])
            response_val=current_feature.convolve_int_image(int_image)
            current_feature.response_vector.append(\
                (row,response_val));
        
        #sort by response value
        current_feature.response_vector.sort(key=lambda tup:tup[1]);
        return;
        #save response matrix as with np.save?                    
    
    def _init_example_weights(self):
        if self._no_pos_training_eg == 0:
            pos_weight=0;
        elif self._no_pos_training_eg == None:
            raise AttributeError("Training Examples not given")
        else:
            pos_weight=0.5/self._no_pos_training_eg;
        
        if self._no_neg_training_eg == 0:
            neg_weight=0;
        elif self._no_neg_training_eg == None:
            raise AttributeError("Training Examples not given")
        else:
            neg_weight=0.5/self._no_neg_training_eg;
        
        for example in self._training_examples:
            if example.label==True:
                example.weight=pos_weight;
            else:
                example.weight=neg_weight;
        return;


    
    def _minimize_threshold_toggle(self,current_feature):
        #if self.example_weights==None:
        #    raise AttributeError('Example weights not defined. Run init_adaboost()');
        
        num_examples=len(self._training_examples);
        toggle=current_feature.toggle;
        
        current_error=current_feature.error;
        current_threshold=current_feature.response_vector[0][1]-1.0;
        current_margin=current_feature.margin;
        
        W_pos_pos=0;
        W_pos_neg=0;#label: positive, detected: negative
        W_neg_pos=0;#label: negative, detected: positive
        W_neg_neg=0;
        
        #since threshold is to the left of smallest example
        #everything is detected as positive at this point.  
          
        for curr_example in self._training_examples:
            if curr_example.label==True:
                W_pos_pos+=curr_example.weight;
            else:
                W_neg_pos+=curr_example.weight;
        
        #print W_pos_pos;
        #print W_neg_pos;
            
        example=1;
            
        while example<=(num_examples+1):
            
            #testing the previous value    
            error_pos=W_pos_neg+W_neg_pos; #error if toggle is +1
            error_neg=W_pos_pos+W_neg_neg; #error if toggle is -1
                
            if(error_neg<error_pos):
                current_error=error_neg;
                toggle=-1;
            else:
                current_error=error_pos;
                toggle=1;
            if(current_error<current_feature.error) or \
              (current_error==current_feature.error and current_margin>current_feature.margin): 
                current_feature.error=current_error;
                current_feature.margin=current_margin;
                current_feature.threshold=current_threshold;
                current_feature.toggle=toggle;
            
            #updating to current value
            
            #if already reached end, don't update
            if example==(num_examples+1):
                break;
             
            #Take care of duplicate response values
            while True:
                #Before: threshold to left of current example
                #After: threshold to right of current example
                
                if self._get_training_labels(example-1,current_feature)==True:
                    W_pos_pos-=self._get_example_weights(example-1,current_feature);
                    W_pos_neg+=self._get_example_weights(example-1,current_feature);
                else:
                    W_neg_neg+=self._get_example_weights(example-1,current_feature);
                    W_neg_pos-=self._get_example_weights(example-1,current_feature);
                    
                                    
                if example==(num_examples) or\
                   current_feature.response_vector[example-1][1]!=current_feature.response_vector[example][1]:
                    break
                else:
                    example+=1;
                
                
            #Update
            if example==(num_examples):
                current_threshold=current_feature.response_vector[example-1][1]+1;
                current_margin=0;
            else:    
                current_threshold=(current_feature.response_vector[example-1][1]+\
                                   current_feature.response_vector[example][1])/2;
                current_margin=current_feature.response_vector[example][1]-\
                               current_feature.response_vector[example-1][1];
            example+=1;
            
            
        #loop ends  
        return;
                    
    
    def _calc_logloss(self):
        """
        Calculates the logarithm of the loss.
        This must minimize with each training round
        This is the most important test of adaboost
        """       
        loss_thisround=0;
        loss_thisexample=0;
        training_error_thisround=0;
        
        if self._no_pos_training_eg == 0:
            pos_weight=0;
        elif self._no_pos_training_eg == None:
            raise AttributeError("Training Examples not given")
        else:
            pos_weight=0.5/self._no_pos_training_eg;
        
        if self._no_neg_training_eg == 0:
            neg_weight=0;
        elif self._no_neg_training_eg == None:
            raise AttributeError("Training Examples not given")
        else:
            neg_weight=0.5/self._no_neg_training_eg;
        
        
        for thisexample in self._training_examples:
            loss_thisexample=0;
            
            if thisexample.label==True:
                yi=1;
            else:
                yi=-1;
            for committee in range(len(self.adaboost_rule)):
                [alpha_t,feature]=self.adaboost_rule[committee];
                decision_stump=(feature.convolve_int_image(thisexample.image)-feature.threshold)*feature.toggle;
                if(decision_stump>0):
                    feature_decision=1;#h(t)->[-1,1]
                else:
                    feature_decision=-1;
                loss_thisexample+=yi*alpha_t*feature_decision;#syi*um (alpha_t*h(t))
            loss_thisround-=loss_thisexample;
            if loss_thisexample<0:
                if thisexample.label == True:
                    init_weight=pos_weight;
                else:
                    init_weight=neg_weight;
                    
                training_error_thisround+=init_weight;
                
        self.log_loss.append(loss_thisround);
        self.training_error.append(training_error_thisround);
        return;
        
                
    
    
    def _get_best_decision_stump(self):
        """
        gets the feature that makes the best decision
        minimizes threshold and toggle to find the best feature
        @retval: min_feature: Feature with minimum error
        """
        min_error=2;
        min_feature=None;
        max_margin=0;
        
        for current_feature in self._features:
            self._minimize_threshold_toggle(current_feature);
            if(current_feature.error<min_error) or (current_feature.error==min_error and current_feature.margin>max_margin):
                min_error=current_feature.error;
                min_feature=current_feature;
                max_margin=current_feature.margin;
        return min_feature;
        
            
    
    def _update_weights(self,bestfeature):
        """
        this function updates the weights of examples after each iteration of adaboost.
        examples misclassified by best feature of the round are given larger weight.
        modifies:self.example_weights
        """                    
        
        tot_examples=len(bestfeature.response_vector);
        
        for example in range(tot_examples):
            actual_eg_no=bestfeature.response_vector[example][0];#example no as saved in example_weights
            if self.weak_detection(bestfeature,example)==True:
                self._training_examples[actual_eg_no].weight=self._training_examples[actual_eg_no].weight*\
                                             0.5*(1.0/(1.0-bestfeature.error));
            else:
                
                self._training_examples[actual_eg_no].weight=self._training_examples[actual_eg_no].weight*\
                                             0.5*(1.0/(bestfeature.error));
                
                
        #Normalize??
        #logging.debug(self.example_weights);
        return;
    
    def weak_detection(self,current_feature,example_no):
        """
        this function returns whether the prediction from single weak detector
        matches the training label or not.
        @retval:True if detector prediction matches training label
        Otherwise False
        """
        prediction=current_feature.toggle\
                   *(current_feature.response_vector[example_no][1]-current_feature.threshold);
        label=self._get_training_labels(example_no,current_feature);
        
        if prediction>0 and label==True:
            return True;
        elif prediction<0 and label==False:
            return True;
        else:
            return False;
        
    
    def _get_training_labels(self,exampleno,current_feature):
        
        actual_eg_no=current_feature.response_vector[exampleno][0];
        return self._training_examples[actual_eg_no].label;
    
    def _get_example_weights(self,exampleno,current_feature):
        
        actual_eg_no=current_feature.response_vector[exampleno][0];
        return self._training_examples[actual_eg_no].weight;
        
    def _reset_features(self):
        for current_feature in self._features:
            current_feature.error=2.0;
            current_feature.threshold=0.;
            current_feature.toggle=1;
            current_feature.margin=0;
        return;
    
        


        
                    
