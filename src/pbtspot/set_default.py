#Always run this before other scripts to give default values to global variables
#ignore at your own peril
#do not change default values here
#change values in specific scripts
#these are default values and should NOT be touched


import globalconstants

def set_default_constants():
    default_const=globalconstants.GlobalConstants();
    #write all default values
    default_const.particle_shape='round'
    default_const.global_window_width=10;
    default_const.global_window_height=10;
    default_const.image_type='A'
    default_const.maxdepth_tree=4;
    default_const.no_pos_training_eg=10;
    default_const.no_neg_training_eg=10;
    default_const.no_pos_testing_eg=10;
    default_const.no_neg_testing_eg=10;
    default_const.prob_threshold=0.95;
    default_const.overfitting_e=0.1;
    default_const.training_rounds=7;
    default_const.is_lowres=False;
    default_const.lowres_width=128;
    default_const.lowres_height=128;
    default_const.gamma=1.

    #default values end here
    default_const.savefile();
    print "Set Default values"
    return;
