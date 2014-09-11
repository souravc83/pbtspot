#global module imports
from __future__ import division
import numpy as np
#local module imports


class ConfusionMatrix(object):
    """
    Finds the confusion matrix.
    public methods:
    def __init__(self)
    def add_to_matrix(self,example,detected_value)
    """
    
    def __init__(self):
        self._true_positive=0;
        self._false_negative=0;
        self._true_negative=0;
        self._false_positive=0;
        
    def add_to_matrix(self,example,detected_value):
        """
        @param: example: Class Example
                detected_val: Boolean
        """
        #first row
        if detected_value==True:
            if example.label==True:
                self._true_positive+=1;
            else:
                self._false_positive+=1;
        else:
            if example.label==True:
                self._false_negative+=1;
            else:
                self._true_negative+=1;
        return;
    
    def get_precision_recall(self):
        try:
            precision=self._true_positive/(self._true_positive+self._false_positive);
        except ZeroDivisionError:
            #this can happen, when we predict all examples to be negative
            # so don't raise an error here
            precision=np.nan;
            #print "Zero denominator in precision calculation."
            
        
        try:
            recall=self._true_positive/(self._true_positive+self._false_negative);
        except ZeroDivisionError:
            #this should not happen if there are any positive examples
            print "Zero denominator in recall calculation."
            print "Does the training set contain no positive examples?"
            raise
        
        return [precision,recall];
    
    def get_tpr_fpr(self):
        
        try:
            false_pos_rate=self._false_positive/(self._false_positive+self._true_negative);
        except ZeroDivisionError:
            print "Zero denominator in false positive rate calculation."
            raise 
        
        try:
            true_pos_rate=self._true_positive/(self._true_positive+self._false_negative);
        except ZeroDivisionError:
            print "Zero denominator in true positive rate calculation."
            raise
        
        return [true_pos_rate,false_pos_rate];
    
    def print_confusion_matrix(self):
        firstline= "Confusion Matrix: "
        secondline= str(self._true_positive)+","+str(self._false_positive);
        thirdline=str(self._false_negative)+","+str(self._true_negative);
        
        return firstline+'\n'+secondline+'\n'+thirdline+'\n'
    
    def get_testing_accuracy(self):
        total_eg=self._true_positive+self._false_positive+\
                 self._false_negative+self._true_negative;
        return (self._true_positive+self._true_negative)/total_eg;
        
        
    
    @property
    def true_positive(self):
        return self._true_positive;
    @property
    def true_negative(self):
        return self._true_negative;
    @property
    def false_positive(self):
        return self._false_positive;
    @property
    def false_negative(self):
        return self._false_negative;
    