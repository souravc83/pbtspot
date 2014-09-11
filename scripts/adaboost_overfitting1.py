"""
this script plots the training and testing accuracy
of Adaboost with the number of training rounds.
"""

from pbtspot import set_default,adaboost_train_test, globalconstants
from pbtspot.adaboost.generateimage import vid_utils
import matplotlib.pyplot as plt

set_default.set_default_constants();
def init_set(glob_const):
    
    #set constants
    glob_const.no_pos_training_eg=100;
    glob_const.no_neg_training_eg=100;
    glob_const.no_pos_testing_eg=1000;
    glob_const.no_neg_testing_eg=1000;
    glob_const.savefile();



def training_round_script():
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    SNRval=0.5;

    #training_round_list=[];
    training_round_list=range(2,15);
    training_acc=[];
    testing_acc=[];
    rounds_tillnow=[];
    pos_tr_eg=glob_const.no_pos_training_eg;
    neg_tr_eg=glob_const.no_neg_training_eg;
    pos_te_eg=glob_const.no_pos_testing_eg;
    neg_te_eg=glob_const.no_neg_testing_eg;


    for training_round in training_round_list:
        boost=adaboost_train_test.AdaBoostTrainTest(SNRval,training_round,\
              pos_tr_eg,neg_tr_eg,pos_te_eg,neg_te_eg);
     
        boost.train_adaboost();
        boost.test_adaboost();
        training_acc.append(boost.training_accuracy);
        testing_acc.append(boost.testing_accuracy);
        rounds_tillnow.append(training_round);
        #this now saves the file at every iteration
        #if you want to quit the program before it finishes
        vid_utils.savefile(rounds_tillnow,training_acc,'data/adaboost_training_acc.txt')
        vid_utils.savefile(rounds_tillnow,testing_acc,'data/adaboost_testing_acc.txt')


    plt.plot(training_round_list,training_acc,'bo-',label='Training Accuracy');
    plt.plot(training_round_list,testing_acc,'ro-',label='Testing Accuracy');
    plt.legend();
    plt.show();

if __name__ == "__main__":
    training_round_script()
