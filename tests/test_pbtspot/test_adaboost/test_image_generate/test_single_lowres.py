"""
@author: Sourav Chatterjee
@brief: tests image_generate.py
"""
#std module imports
import nose.tools

#local module imports
from src.pbtspot.adaboost.generateimage import image_generate,single_lowres
reload(image_generate)
reload(single_lowres)

class TestSingleLowres:
    def setup(self):
        self.SNR=10;
        self.genimg=single_lowres.SingleLowres(self.SNR);
        
    def test_generate_particles(self):
        self.genimg.generate_noise();
        self.genimg.generate_particles();
        nose.tools.assert_equal(len(self.genimg.particle_loc),1);
        print "particle_loc: "+str(self.genimg.particle_loc)
        nose.tools.assert_true(self.genimg.particle_loc[0][0]<self.genimg.width);
        nose.tools.assert_true(self.genimg.particle_loc[0][1]<self.genimg.height);
        return;
    
    def test_generate_img(self):
        self.genimg.generate_img();
        self.genimg.show_image('Noisy');
        return;
    
    def test_lowres_image(self):
        self.genimg.lowres_image();
        self.genimg.show_image('Lowres');
        return;
        
        
    
