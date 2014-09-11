"""
This script varies overfitting_e, keeping the training_rounds constant
It then calculates the training and testing accuracies
"""
from pbtspot import set_default,train_and_test, globalconstants
from pbtspot.adaboost.generateimage import vid_utils
import matplotlib.pyplot as plt


def init_set(glob_const):
    set_default.set_default_constants();
    
    #set constants
    glob_const.no_pos_training_eg=500;
    glob_const.no_neg_training_eg=500;
    glob_const.no_pos_testing_eg=4000;
    glob_const.no_neg_testing_eg=4000;
    glob_const.training_rounds=7;
    glob_const.maxdepth_tree=4;
    glob_const.savefile();



def training_round_script():
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    SNRval=2.;

    #overfitting_e_list=[];
    overfitting_e_list=[0.1*x for x in range(0,5)];
    training_acc=[];
    testing_acc=[];
    list_till_now=[]
    
    for overfitting_e in overfitting_e_list:
        glob_const.overfitting_e=overfitting_e;
        glob_const.savefile();
        traintest=train_and_test.TrainAndTest(SNRval);
        traintest.train_tree();
        traintest.test_tree();
        list_till_now.append(overfitting_e)
        training_acc.append(traintest.training_accuracy);
        testing_acc.append(traintest.testing_accuracy);
        vid_utils.savefile(list_till_now,training_acc,'data/overfittng_e_training_acc.txt')
        vid_utils.savefile(list_till_now,testing_acc,'data/overfitting_e_testing_acc.txt')
        print "Processed e: "+str(overfitting_e)+"...."
        
    plt.plot(overfitting_e_list,training_acc,'bo-',label='Training Accuracy');
    plt.plot(overfitting_e_list,testing_acc,'ro-',label='Testing Accuracy');
    plt.legend();
    plt.show();






    
    
    
    
    pass

if __name__ == "__main__":
    training_round_script()
