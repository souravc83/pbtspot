"""
@package: test_features
@author:Sourav Chatterjee
"""
#import from std modules
import nose.tools
import numpy as np

#import from local modules
from src.pbtspot.adaboost.features import diff_features,basefeature
from src.pbtspot.adaboost import currimage
reload(diff_features)
reload(basefeature)

from src.pbtspot.adaboost import currimage
reload(currimage)

class TestFeatures:
    def setup(self):
        #image with all pixels same
        self.eg_img=np.ones((10,10));
    
    #def teardown(self):
    #    print "TEAR DOWN!"

    def get_int_image(self,img):
        currimg=currimage.CurrImage(img);
        currimg.calc_integral_image();
        int_image=currimg.int_image;
        return int_image
 
    def test_Edge_Features(self):
        #Edge Feature 1
        Edge1=diff_features.EdgeFeature_1();
        Edge1.generate_feature(0,0,4,1);
        [r1,r2]=Edge1.rectlist;
        nose.tools.assert_equal(r2.x_val,2);
        nose.tools.assert_equal(r2.y_val,0);
        nose.tools.assert_equal(r2.width,2);
        
        px_sum=Edge1.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum,0);
        #Edge1.show_feature();
        
        #change to image with unequal pixel values
        self.eg_img[:,0:2]=0;
        px_sum1=Edge1.convolve_int_image(self.get_int_image(self.eg_img))
        nose.tools.assert_equal(px_sum1,-2);       
        
        #reset
        self.eg_img[:,0:2]=1;
        
        #Edge Feature 2
        Edge2=diff_features.EdgeFeature_2();
        Edge2.generate_feature(0,0,1,4);
        [r3,r4]=Edge2.rectlist;
        nose.tools.assert_equal(r4.x_val,0);
        nose.tools.assert_equal(r4.y_val,2);
        nose.tools.assert_equal(r4.height,2);
        
        px_sum=Edge2.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum,0);
        #Edge2.show_feature();
        
        #unequal pixel values
        self.eg_img[0:2,]=0;
        px_sum1=Edge2.convolve_int_image(self.get_int_image(self.eg_img))
        nose.tools.assert_equal(px_sum1,-2);       
        
        #reset
        self.eg_img[0:2,:]=1;
        
    def test_Line_Features(self):
        #Line Feature 1
        Line1=diff_features.LineFeature_1();
        Line1.generate_feature(0,0,6,4);
        [r1,r2,r3]=Line1.rectlist;
        nose.tools.assert_equal(r2.x_val,2);
        nose.tools.assert_equal(r3.x_val,4);
        
        px_sum=Line1.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum,0);
        #Line1.show_feature();
        
        #unequal
        self.eg_img[:,0:2]=0;
        px_sum1=Line1.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum1,8);       
        
        #reset
        self.eg_img[:,0:2]=1;
        
        #Line Feature 2
        Line2=diff_features.LineFeature_2();
        Line2.generate_feature(0,0,4,6);
        [r4,r5,r6]=Line2.rectlist;
        nose.tools.assert_equal(r5.y_val,2);
        nose.tools.assert_equal(r6.y_val,4);
        
        px_sum=Line2.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum,0);
        
        #Line2.show_feature();
        
        
        #unequal
        self.eg_img[0:2,:]=0;
        px_sum1=Line2.convolve_int_image(self.get_int_image(self.eg_img))
        nose.tools.assert_equal(px_sum1,8);       
        
        #reset
        self.eg_img[0:2,:]=1;
            
        #Line Feature 3
        Line3=diff_features.LineFeature_3();
        Line3.generate_feature(0,0,8,3);
        [r7,r8,r9]=Line3.rectlist;
        nose.tools.assert_equal(r8.x_val,2);
        nose.tools.assert_equal(r9.x_val,6);
        
        px_sum=Line3.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum,0);
        #Line3.show_feature();
        
        
        #unequal
        self.eg_img[:,2:6]=0;
        px_sum1=Line3.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum1,-12);       
        
        #reset
        self.eg_img[:,2:6]=1;
        
        #Line Feature 4
        Line4=diff_features.LineFeature_4();
        Line4.generate_feature(0,0,3,8);
        [r10,r11,r12]=Line4.rectlist;
        nose.tools.assert_equal(r11.y_val,2);
        nose.tools.assert_equal(r12.y_val,6);
        px_sum=Line4.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum,0);
        #Line4.show_feature();
        
        #unequal
        self.eg_img[2:6,:]=0;
        px_sum1=Line4.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum1,-12);       
        
        #reset
        self.eg_img[2:6,:]=1;
    
    def test_center_Features(self): 
        #Center Feature
        Center1=diff_features.CenterFeature();
        Center1.generate_feature(0,0,3,6);
        [r1,r2,r3,r4,r5]=Center1.rectlist;
        nose.tools.assert_equal(r5.y_val,2);
        nose.tools.assert_equal(r4.x_val,2);
        
        px_sum=Center1.convolve_int_image(self.get_int_image(self.eg_img));
        nose.tools.assert_equal(px_sum,0);
        Center1.show_feature();
        
        #unequal
        self.eg_img[2:4,1]=0;
        px_sum1=Center1.convolve_int_image(self.get_int_image(self.eg_img))
        #nose.tools.assert_equal(px_sum1,-16);       
        
        #reset
        self.eg_img[2:4,1]=1;
        
        #for rect in Center1.rectlist:
        #    print rect;
            
        
        
