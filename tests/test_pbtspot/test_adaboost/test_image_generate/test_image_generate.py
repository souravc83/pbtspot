"""
@author: Sourav Chatterjee
@brief: tests image_generate.py
"""
#std module imports
import nose.tools

#local module imports
from src.pbtspot.adaboost.generateimage import image_generate
reload(image_generate)

class TestGenerateImage:
    
    def setup(self):
        self.SNR=10;
        self.genimg=image_generate.GenerateImage(self.SNR);

    def test_generate_img(self):
        self.genimg.part_rows=2;
        self.genimg.part_columns=2;
        self.genimg.generate_img();
        
        for i in range(2):
            x_loc=self.genimg.particle_loc[i][0];
            y_loc=self.genimg.particle_loc[i][1];
            intensity=self.genimg.imgarr[y_loc,x_loc]
            print intensity
            #nose.tools.assert_true(intensity>100);#reasonably bright, change this later
        return;
        
            
        