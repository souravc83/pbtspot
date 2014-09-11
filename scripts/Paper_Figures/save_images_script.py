"""
This script extracts images of different types and saves them.
This is for Figure 1 in the paper
"""

#global module imports
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#local imports
from pbtspot import set_default,globalconstants
from pbtspot.adaboost.generateimage import image_generate,vid_utils


set_default.set_default_constants();

def init_set(glob_const):
    #set constants
    glob_const.particle_shape='round'
    glob_const.image_type='A'
    glob_const.savefile();

def gen_img_type(shape,imagetype, SNRval,filename,islowres=False):

    glob_const=globalconstants.GlobalConstants();
    init_set(glob_const);
    
    glob_const.particle_shape=shape;
    glob_const.image_type=imagetype;
    glob_const.savefile();

    img_gen=image_generate.GenerateImage(SNRval);
    if islowres==False:
        img_gen.generate_img();
        full_img_matrix=img_gen.noisy_imgarr;
        part_img_matrix=full_img_matrix[0:150,0:130];
    else:
        img_gen.lowres_image();
        full_img_matrix=img_gen.noisy_lowres;
        part_img_matrix=full_img_matrix[0:37,0:32];
        
    #filename='scripts/Paper_Figures/Figure1/A_round_SNR2.png';
    
    plt.imshow(part_img_matrix,cmap=cm.Greys_r,interpolation='none');
    plt.axis('off')
    #plt.show()
    #also tried scipy.misc.imsave but this provides more 
    #flexibility and more options
    plt.savefig(filename,dpi=300);
    plt.show();
    return;

def imggen_script():
    
    #filename='scripts/Paper_Figures/Figure1/A_round_SNR2.png';
    #gen_img_type(shape='round',imagetype='A',SNRval=2.,\
    #             filename=filename);
    #filename='scripts/Paper_Figures/Figure1/A_round_SNR0p5.png';
    #gen_img_type(shape='round',imagetype='A',SNRval=0.5,\
    #             filename=filename);
    #filename='scripts/Paper_Figures/Figure1/A_elongated_SNR2.png';
    #gen_img_type(shape='elongated',imagetype='A',SNRval=2,\
    #             filename=filename);   
    #filename='scripts/Paper_Figures/Figure1/B_elongated_SNR2.png';
    #gen_img_type(shape='elongated',imagetype='B',SNRval=2,\
    #             filename=filename);    
    filename='scripts/Paper_Figures/Figure1/A_round_SNR2_lowres.png';
    gen_img_type(shape='round',imagetype='A',SNRval=2,\
                 filename=filename,islowres=True);
    filename='scripts/Paper_Figures/Figure1/A_round_SNR0p5_lowres.png';
    gen_img_type(shape='round',imagetype='A',SNRval=0.5,\
                 filename=filename,islowres=True);
    
    return;



if __name__=="__main__":
    imggen_script()

