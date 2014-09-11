"""
this script plots training and testing accuracy of tree with 
depth.
"""

#global imports

#local imports
from pbtspot import set_default,train_and_test, globalconstants


set_default.set_default_constants();
def init_set(glob_const):
    
    #set constants
    glob_const.no_pos_training_eg=400;
    glob_const.no_neg_training_eg=400;
    glob_const.no_pos_testing_eg=4000;
    glob_const.no_neg_testing_eg=4000;
    glob_const.maxdepth_tree=3;
    glob_const.training_rounds=7;
    glob_const.prob_threshold=0.5
    glob_const.savefile();



def depth_script():
    #important: line 199 in function train_node.py 
    #which is self._print_histogram in create_children()
    #function is uncommented for this script to run
    #comment it back and reinstall after the script runs
    
    #this also depends on information from self._print_loginfo
    #in tree_node.py
    
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    SNRval=0.5;

    traintest=train_and_test.TrainAndTest(SNRval);
    traintest.train_tree();
    
if __name__=="__main__":
    depth_script()
