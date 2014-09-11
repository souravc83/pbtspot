"""
Tests overfitting in the tree by plotting training and 
testing accuracy as a function of training rounds
"""
from pbtspot import set_default, globalconstants
from pbtspot.adaboost import trainer,example_images
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
    training_round_list=range(2,11);
    training_acc=[];
    testing_acc=[];
    
    for training_round in training_round_list:
        glob_const.training_rounds=training_round;
        glob_const.savefile();
        traintest=train_and_test.TrainAndTest(SNRval);
        traintest.train_tree();
        traintest.test_tree();
        training_acc.append(traintest.training_accuracy);
        testing_acc.append(traintest.testing_accuracy);

    vid_utils.savefile(training_round_list,training_acc,'data/training_acc.txt')
    vid_utils.savefile(training_round_list,testing_acc,'data/testing_acc.txt')
    plt.plot(training_round_list,training_acc,'bo-',label='Training Accuracy');
    plt.plot(training_round_list,testing_acc,'ro-',label='Testing Accuracy');
    plt.legend();
    plt.show();






    
    
    
    
    pass

if __name__ == "__main__":
    training_round_script()
