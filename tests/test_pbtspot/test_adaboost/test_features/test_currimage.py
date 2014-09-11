"""
@package: test_currimage
@author:Sourav Chatterjee
@brief: test currimage.py
"""
#import from std modules
import nose.tools
import numpy as np

#import from local modules
from src.pbtspot.adaboost import currimage
reload(currimage)

class TestCurrImage:
    
    def setup(self):
        self.eg_img=np.ones((3,3));
        self.eg_int_img=np.array([(1,2,3),
                                   (2,4,6),
                                   (3,6,9)]);
    
    def test_calc_integral_image(self):
        currimg=currimage.CurrImage(self.eg_img);
        currimg.calc_integral_image();
        int_img=currimg.int_image;
        np.testing.assert_array_equal(int_img,self.eg_int_img);