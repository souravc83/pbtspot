#import from std library
from __future__ import division
import copy

#import from local modules
from features import basefeature,diff_features
import currimage

class Detector:
    """
    given an adaboost rule, make a detection
    """
    
    def __init__(self,adaboost_rule):
        self.adaboost_rule=adaboost_rule
        
    def get_decision_stump(self,thisexample):
        """
        given an example returns the decision stump
        sum (alpha_t*h(t)) for all training rounds t.
        The sign of the decision stump is the prediction.
        To simply get the sign of the decision stump, use
        get_binary_decision(thisexample)
        
        @param: thisexample: instance of class Example
        @retval: decision_value: float whose sign gives True or False decision
        
        """
        
        tot_training_rounds=len(self.adaboost_rule);
        decision_value=0.; #sum(alpha_t*h(t))
        decision_thisround=0.;
        currimg=currimage.CurrImage(thisexample.image);
        currimg.calc_integral_image();
        base_int_image=currimg.int_image;

        for training_round in range(tot_training_rounds):
            [alpha_t,feature]=self.adaboost_rule[training_round];
            int_image=copy.deepcopy(base_int_image);

            decision_stump=(feature.convolve_int_image(int_image)-feature.threshold)*feature.toggle;
            if(decision_stump>0):
                decision_thisround=1;#h(t)->[-1,1]
            else:
                decision_thisround= -1;
            decision_value+=alpha_t*decision_thisround;#(alpha_t*h(t))
        
        return decision_value;
    
    
    def get_binary_decision(self,thisexample):
        """
        @param: thisexample: instance of class Example
        @retval: True or False, signifying whether example is positive or negative
        """
        
        decision_value=self.get_decision_stump(thisexample); #think about caching this value
        #so that you don't end up calculating the same decision for two 
        #examples.
        
        #you'll have to decide what to do with 0 value??
        
        if decision_value>0.:
            return True;
        else:
            return False;
            
            
