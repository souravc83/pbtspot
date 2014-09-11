"""
@author: Sourav Chatterjee
@brief: tests trainer
"""

#import from std modules
import nose.tools
import logging
import numpy as np
import matplotlib.pyplot as plt

#import local modules
from src.pbtspot.adaboost import trainer,example_images,strong_detector,currimage
reload(trainer)
reload(example_images)
from src.pbtspot.adaboost.features import diff_features,basefeature
reload(diff_features)
from src.pbtspot.adaboost.generateimage import sample_generate

class TestTrainer:
    def setup(self):
        self.trainer=trainer.Trainer(4);
        self.SNR=10;
        logging.basicConfig(filename='logfile.txt',level=logging.DEBUG,filemode='w')
    
    def get_training_examples(self):
        
        no_pos_eg=10;
        no_neg_eg=10;
        eg_imgs=example_images.ExampleImages(self.SNR,no_pos_eg,no_neg_eg);#should this be self?
        training_examples=eg_imgs.load_examples('Training');
        
        self.trainer.training_examples=training_examples;
        
        return;
    
    def test_generate_features(self):
        #These are the feature numbers taken directly 
        #from Table 1 of the paper:
        #Lienhart and Mayt
        #"An extended set of Haar like Features for Rapid Object Detection"
        self.trainer.global_window_width=24;
        self.trainer.global_window_height=24;
        
        self.trainer._features=[];
        
        #Edge features
        Edge1=diff_features.EdgeFeature_1(); 
        self.trainer._generate_each_feature(Edge1);
        nose.tools.assert_equal(self.trainer.test_featurecount,43200);
        
        Edge2=diff_features.EdgeFeature_2(); 
        self.trainer._generate_each_feature(Edge2);
        nose.tools.assert_equal(self.trainer.test_featurecount,43200);
        
        #Line Features
        Line1=diff_features.LineFeature_1(); 
        self.trainer._generate_each_feature(Line1);
        nose.tools.assert_equal(self.trainer.test_featurecount,27600);
        
        Line2=diff_features.LineFeature_2(); 
        self.trainer._generate_each_feature(Line2);
        nose.tools.assert_equal(self.trainer.test_featurecount,27600);
        
        #Todo: These two don't agree with the paper. Find out why?
        Line3=diff_features.LineFeature_3(); 
        self.trainer._generate_each_feature(Line3);
        nose.tools.assert_equal(self.trainer.test_featurecount,19800);
        
        Line4=diff_features.LineFeature_4(); 
        self.trainer._generate_each_feature(Line4);
        nose.tools.assert_equal(self.trainer.test_featurecount,19800);
        
        Center1=diff_features.CenterFeature();
        self.trainer._generate_each_feature(Center1);
        nose.tools.assert_equal(self.trainer.test_featurecount,8464);
        
        
        #set xlim and ylim back for further tests
        self.trainer.global_window_width=10;
        self.trainer.global_window_height=10;
        
        
    def test_calc_response_vector(self):
        self.trainer._features=[];
        Edge1=diff_features.EdgeFeature_1();
        self.trainer._generate_each_feature(Edge1);
        single_feature=self.trainer._features[0]; 
        self.get_training_examples();
        self.trainer._int_image_list=[]

        for example in self.trainer._training_examples:
            currimg=currimage.CurrImage(example.image);
            currimg.calc_integral_image();
            int_image=currimg.int_image;
            self.trainer._int_image_list.append(int_image);
            
        self.trainer._calc_response_vector(single_feature);
        
        
    def test_run_adaboost(self):
    
        self.trainer.adaboost_rule=None;
        
      
        
        self.get_training_examples();
        adaboost_rule=self.trainer.run_adaboost();
        
        print self.trainer._no_neg_training_eg;
        for i in range(len(adaboost_rule)):
            [alpha,feature]=adaboost_rule[i];
            print "threshold: "+str(feature.threshold);
            print "error: "+str(feature.error);
        
        [alpha,feature]=adaboost_rule[0];
        logging.debug(str(feature.threshold));
    
    def test_trainer_messy_example(self):
        mytrainer=trainer.Trainer(10);
        eg_imgs=example_images.ExampleImages(0.5,30,30);#should this be self?
        training_examples=eg_imgs.load_examples('Training');
        mytrainer.training_examples=training_examples;
        adaboost_rule=mytrainer.run_adaboost();
        
        print "Hard example. should have more than one rounds"
    
        for i in range(len(adaboost_rule)):
            [alpha,feature]=adaboost_rule[i];
            print "threshold: "+str(feature.threshold);
            print "error: "+str(feature.error);
            print "toggle: "+str(feature.toggle);
        
        #test the strong detector
        testing_eg=eg_imgs.load_examples('Testing');
        detector=strong_detector.Detector(adaboost_rule);
        
        print "Posterior Probabilities for examples at low SNR: "
        poslist=[];
        neglist=[];
        for example in testing_eg:
            decision=detector.get_decision_stump(example);
            q_pos=1./(1.+np.exp(-2*decision));
            if example.label==True:
                poslist.append(q_pos)
                #print "True: "+str(q_pos);
            else:
                neglist.append(q_pos)
                #print "False:"+str(q_pos);
        plt.figure();
        plt.hist(np.array(poslist),bins=10,range=(0,1),color='r',label='Positive')
        
        plt.hist(np.array(neglist),bins=10,range=(0,1),color='b',label='Negative')
        plt.xlabel("Posterior Probability")
        plt.ylabel("Number of Examples")
        plt.legend()
        plt.title('Low SNR')
        plt.show()
        return
        
    
    def rect_compare(self,rect1,rect2):
        """
        custom function to compare Rectangle insances
        """
        nose.tools.assert_equal(rect1.x_val,rect2.x_val);
        nose.tools.assert_equal(rect1.y_val,rect2.y_val);
        nose.tools.assert_equal(rect1.width,rect2.width);
        nose.tools.assert_equal(rect1.height,rect2.height);
    
    
    def test_trainer_centerfeature(self):
        mytrainer=trainer.Trainer(10);
        mysample=sample_generate.TestImages();
        mysample.gen_center_feature();
        pos_eg=10;
        neg_eg=10;
        this_pos_example=example_images.Example(mysample.noisy_imgarr,True,float(1./pos_eg));
        mysample.reset_image();
        mysample.gen_neg_blank();
        this_neg_example=example_images.Example(mysample.noisy_imgarr,False,float(1./neg_eg));
        training_examples=[this_pos_example]*pos_eg;
        training_examples.extend([this_neg_example]*neg_eg);
        
        mytrainer.training_examples=training_examples;
        mytrainer.run_adaboost();
        [alpha,foundfeature]=mytrainer.adaboost_rule[0];
        nose.tools.assert_equal(foundfeature.__repr__(),"CenterFeature")
        #nose.tools.assert_equal(foundfeature.rectlist[0],mysample.firstrect)
        self.rect_compare(foundfeature.rectlist[0],mysample.firstrect)         
        
