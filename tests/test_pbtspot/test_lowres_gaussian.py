#import from std modules
import nose.tools

#import from local modules
from src.pbtspot import lowres_gaussian_fit,boosting_tree,train_and_test


class TestLowresGaussian:
    
    def setup(self):
        self.SNR=10.;
        self.lowgauss=lowres_gaussian_fit.LowresGaussianFit(self.SNR);
        self.lowgauss.no_pos_training_eg=10;
        self.lowgauss.no_neg_training_eg=10;
        self.lowgauss.train_tree();
        
    
    def test_generate_samples(self):
         self.lowgauss.generate_matrix();
         print self.lowgauss.prob_matrix;
