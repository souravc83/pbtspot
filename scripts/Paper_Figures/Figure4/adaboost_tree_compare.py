from pbtspot import set_default,train_and_test, globalconstants,\
                    adaboost_train_test
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
    glob_const.maxdepth_tree=5;
    glob_const.savefile();



def compare_script():
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    #SNRlist=[];
    SNRlist=[0.5*x for x in range(1,2)];
    tree_acc=[];
    boost_acc=[];


    training_rounds=1;
    pos_tr_eg=glob_const.no_pos_training_eg;
    neg_tr_eg=glob_const.no_neg_training_eg;
    pos_te_eg=glob_const.no_pos_testing_eg;
    neg_te_eg=glob_const.no_neg_testing_eg;
    list_till_now=[];

    
    for SNRval in SNRlist:
        boost=adaboost_train_test.AdaBoostTrainTest(SNRval,training_rounds,\
              pos_tr_eg,neg_tr_eg,pos_te_eg,neg_te_eg);
        traintest=train_and_test.TrainAndTest(SNRval);
        
        #Train
        boost.train_adaboost();
        traintest.train_tree();

        #Test
        boost.test_adaboost();
        traintest.test_tree();

        #get accuracies
        tree_acc.append(boost.testing_accuracy);
        boost_acc.append(traintest.testing_accuracy);
        list_till_now.append(SNRval);
        vid_utils.savefile(list_till_now,tree_acc,"data/Tree_accuracies.txt")
        vid_utils.savefile(list_till_now,boost_acc,"data/Boost_accuracies.txt")
        
        
    plt.plot(SNRlist,tree_acc,'ro',label="Tree");
    plt.plot(SNRlist,boost_acc,'bo',label="Adaboost");
    plt.xlim(0, 3)
    plt.legend();
    plt.show();

    
    


if __name__ == "__main__":
    compare_script()
