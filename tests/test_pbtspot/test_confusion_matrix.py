"""
@author: Sourav Chatterjee
@brief: tests confusion_matrix.py
"""

#import from std modules
import nose.tools

#import from local modules
from src.pbtspot import confusion_matrix
from src.pbtspot.adaboost import example_images

class TestConfusionMatrix:
    
    def setup(self):
        self.conf_mat=confusion_matrix.ConfusionMatrix();
    
    def test_add_to_matrix(self):
        eg_img1=example_images.ExampleImages(10,1,1);
        example_list=eg_img1.load_examples('Training');
        pos_example=example_list[0];
        self.conf_mat.add_to_matrix(pos_example,False);
        nose.tools.assert_equal(self.conf_mat._false_negative,1)
        self.conf_mat.add_to_matrix(pos_example,True);
        nose.tools.assert_equal(self.conf_mat._true_positive,1)
        neg_example=example_list[1];
        self.conf_mat.add_to_matrix(neg_example,True);
        nose.tools.assert_equal(self.conf_mat._false_positive,1)
        self.conf_mat.add_to_matrix(neg_example,False);
        nose.tools.assert_equal(self.conf_mat._true_negative,1)
        return;
    
    def test_get_precision_recall(self):
        
        eg_img1=example_images.ExampleImages(10,1,1);
        example_list=eg_img1.load_examples('Training');
        self.conf_mat.add_to_matrix(example_list[0],False); #Wrong value
        self.conf_mat.add_to_matrix(example_list[1],True); #wrong value
        [precision,recall]=self.conf_mat.get_precision_recall();
        nose.tools.assert_equal(precision,0);
        nose.tools.assert_equal(recall,0);
        [tpr,fpr]=self.conf_mat.get_tpr_fpr();
        nose.tools.assert_equal(tpr,0);
        nose.tools.assert_equal(fpr,1);
        
        
        
        
        
