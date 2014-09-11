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
    glob_const.no_pos_training_eg=100;
    glob_const.no_neg_training_eg=100;
    glob_const.no_pos_testing_eg=4000;
    glob_const.no_neg_testing_eg=4000;
    glob_const.maxdepth_tree=4;
    glob_const.training_rounds=7;
    glob_const.prob_threshold=0.5;
    glob_const.gamma=1.;
    glob_const.overfitting_e=0.1;
    glob_const.savefile();



def gamma_script():
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    SNRval=2.;

    #maxdepth_list=[];
    #gamma_list=range(1,7);
    gamma_list=[0];
    training_acc=[];
    testing_acc=[];
    list_tillnow=[];
    time_list=[];
    for gamma in gamma_list:
        starttime=time.time();
        
        glob_const.gamma=gamma;
        glob_const.savefile();
        traintest=train_and_test.TrainAndTest(SNRval);
        traintest.train_tree();
        traintest.test_tree();
        elapsed=time.time()-starttime;
        
        #append values
        training_acc.append(traintest.training_accuracy);
        testing_acc.append(traintest.testing_accuracy);
        list_tillnow.append(gamma);
        time_list.append(elapsed)
        vid_utils.savefile(list_tillnow,training_acc,\
                           'data/training_acc_gamma.txt')
        vid_utils.savefile(list_tillnow,testing_acc,\
                'data/testing_acc_gamma.txt')
        vid_utils.savefile(list_tillnow,time_list,\
                'data/time_vals_depth.txt')

    plt.plot(gamma_list,training_acc,'bo-',label='Training Accuracy');
    plt.plot(gamma_list,testing_acc,'ro-',label='Testing Accuracy');
    plt.legend();
    plt.show();


if __name__=="__main__":
    gamma_script()
