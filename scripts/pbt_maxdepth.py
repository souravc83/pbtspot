"""
this script plots training and testing accuracy of tree with 
depth.
"""

#global imports
import matplotlib.pyplot as plt
import time

#local imports
from pbtspot import set_default,train_and_test, globalconstants
from pbtspot.adaboost.generateimage import vid_utils



def init_set(glob_const):
    set_default.set_default_constants();
    #set constants
    glob_const.no_pos_training_eg=500;
    glob_const.no_neg_training_eg=500;
    glob_const.no_pos_testing_eg=4000;
    glob_const.no_neg_testing_eg=4000;
    glob_const.maxdepth_tree=2;
    glob_const.training_rounds=7;
    glob_const.prob_threshold=0.5
    glob_const.savefile();



def depth_script():
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    SNRval=2.;

    #maxdepth_list=[];
    maxdepth_list=range(2,7);
    training_acc=[];
    testing_acc=[];
    list_tillnow=[];
    time_list=[];
    for depth in maxdepth_list:
        starttime=time.time();
        
        glob_const.maxdepth_tree=depth;
        glob_const.savefile();
        traintest=train_and_test.TrainAndTest(SNRval);
        traintest.train_tree();
        traintest.test_tree();
        traintest.post_prune_tree();
        traintest.test_tree();
        elapsed=time.time()-starttime;
        
        #append values
        training_acc.append(traintest.training_accuracy);
        testing_acc.append(traintest.testing_accuracy);
        list_tillnow.append(depth);
        time_list.append(elapsed)
        vid_utils.savefile(list_tillnow,training_acc,\
                           'data/training_acc_depth.txt')
        vid_utils.savefile(list_tillnow,testing_acc,\
                'data/testing_acc_depth.txt')
        vid_utils.savefile(list_tillnow,time_list,\
                'data/time_vals_depth.txt')

    plt.plot(maxdepth_list,training_acc,'bo-',label='Training');
    plt.plot(maxdepth_list,testing_acc,'ro-',label='Testing');
    plt.legend(loc=2);
    plt.show();


if __name__=="__main__":
    depth_script()
