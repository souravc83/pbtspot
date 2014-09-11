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
    glob_const.maxdepth_tree=5;
    glob_const.savefile();

def roc_script(SNRval,particle_shape,image_type):
    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    
    glob_const.image_type=image_type;
    glob_const.particle_shape=particle_shape;
    glob_const.savefile();
    
 
    
    traintest=train_and_test.TrainAndTest(SNRval);
    traintest.train_tree();
    traintest.test_tree();
    [precision,recall]=traintest.get_precision_recall_curve();
    precision_filename="data/precision_"+str(SNRval)+"_"+str(image_type)+\
                        "_"+str(particle_shape)+".txt";
    vid_utils.savefile(precision,recall,precision_filename)
    [tpr,fpr]=traintest.get_roc_curve();
    roc_filename="data/roc_"+str(SNRval)+"_"+str(image_type)+\
                        "_"+str(particle_shape)+".txt";
    
    vid_utils.savefile(tpr,fpr,roc_filename)
    
def mainscript():
    roc_script(0.5,'round','A');
    roc_script(2.,'round','A');
    roc_script(2.,'elongated','A');
    roc_script(2.,'round','B');
        
    
if __name__ == "__main__":
    mainscript()    
    
    