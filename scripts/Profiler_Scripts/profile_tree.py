"""
this code profiles the runtime of the trainer.py
which trains an Adaboost rule
"""
#import global modules
import cProfile
import pstats
import time

#import local modules

from pbtspot import set_default, globalconstants, train_and_test
#from pbtspot.adaboost.generateimage import vid_utils

set_default.set_default_constants();
def init_set(glob_const):
    
    #set constants
    glob_const.no_pos_training_eg=200;
    glob_const.no_neg_training_eg=200;
    glob_const.no_pos_testing_eg=1000;
    glob_const.no_neg_testing_eg=1000;
    glob_const.maxdepth_tree=3;
    glob_const.training_rounds=7;
    
    glob_const.savefile();



def training_round_script():
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    examples=glob_const.no_pos_training_eg;
    
    init_t=time.time()

    SNRval=0.5;
    traintest=train_and_test.TrainAndTest(SNRval);
    traintest.train_tree();
    print "Trained: "+str(time.time()-init_t)
    init_t=time.time()
    traintest.test_tree();
    print "Tested: "+str(time.time()-init_t)
    init_t=time.time()
    traintest.post_prune_tree();
    print "Post Pruned: "+str(time.time()-init_t)
    
    return;


def profiletrainer():
    cProfile.run('training_round_script()','data/restats');
    stats=pstats.Stats('data/restats');
    stats.strip_dirs().sort_stats('tottime').print_stats()
    return;

if __name__=="__main__":
    profiletrainer()





