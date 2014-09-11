"""
this code profiles the runtime of the trainer.py
which trains an Adaboost rule
"""
#import global modules
import cProfile
import pstats

#import local modules

from pbtspot import set_default, globalconstants
from pbtspot.adaboost import trainer,example_images
#from pbtspot.adaboost.generateimage import vid_utils

set_default.set_default_constants();
def init_set(glob_const):
    
    #set constants
    glob_const.no_pos_training_eg=10;
    glob_const.no_neg_training_eg=10;
    glob_const.no_pos_testing_eg=1000;
    glob_const.no_neg_testing_eg=1000;
    glob_const.savefile();



def training_round_script():
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    examples=glob_const.no_pos_training_eg;

    SNRval=0.5;
    eg_imgs=example_images.ExampleImages(SNRval,examples,examples);
    training_examples=eg_imgs.load_examples('Training');
    mytrainer=trainer.Trainer();
    mytrainer.training_examples=training_examples;
    rule=mytrainer.run_adaboost();
    return;


def profiletrainer():
    cProfile.run('training_round_script()','data/restats');
    stats=pstats.Stats('data/restats');
    stats.strip_dirs().sort_stats('tottime').print_stats()
    return;

if __name__=="__main__":
    profiletrainer()





