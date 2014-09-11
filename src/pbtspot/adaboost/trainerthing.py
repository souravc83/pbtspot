#import std modules
from __future__ import division
import logging
import copy

#import local modules
import trainer
reload(trainer)
from Features import diff_features,basefeature
reload(diff_features)
reload(basefeature)
from Image_Generate import image_generate, sample_generate
reload(image_generate)
reload(sample_generate)

import example_images
reload(example_images)

logging.basicConfig(filename='logfile.txt',level=logging.DEBUG,filemode='w')

def test_trainer1():
     mytrainer=trainer.Trainer();
     mytrainer.no_pos_training_eg=10;
     mytrainer.no_neg_training_eg=10;
     
     
     adaboost_rule=mytrainer.run_adaboost(1);
     
    
     [alpha,feature]=adaboost_rule[9];
     print "threshold: "+str(feature.threshold);
     print "error: "+str(feature.error);
     print "toggle: "+str(feature.toggle);
     print "Initial Weights:"+str(0.5/10)
     response_vector=feature.response_vector;
     response_vector.sort(key=lambda tup:tup[0]);
     tot_examples=len(response_vector);
     prediction=None;
     
     for i in range(tot_examples):
         if(feature.toggle*(response_vector[i][1]-feature.threshold)>0):
             prediction=True;
         else:
             prediction=False;
         
         printstr=str(response_vector[i])+"Label: "+str(mytrainer.training_labels[i])+\
                  " Detected:"+str(prediction)+\
                  " Wt:"+str(mytrainer.example_weights[i]);
                  
         print printstr;
     
     return;

def test_trainer2():
    mytrainer=trainer.Trainer();
    mytrainer.no_pos_training_eg=30;
    mytrainer.no_neg_training_eg=30;
    adaboost_rule=mytrainer.run_adaboost(40);
    
    for i in range(len(adaboost_rule)):
        [alpha,feature]=adaboost_rule[i];
        print "threshold: "+str(feature.threshold);
        print "error: "+str(feature.error);
        print "toggle: "+str(feature.toggle);

def test_trainer3():
    mytrainer=trainer.Trainer();
    mytrainer.no_pos_training_eg=1;
    mytrainer.no_neg_training_eg=2;
    mytrainer.get_training_examples();
    
    mytrainer.example_weights=[0.33,0.33,0.33];
    mytrainer.training_labels=[True, False,False];
    mytrainer.generate_features();
    feature=copy.deepcopy(mytrainer.features[0]);
    feature.response_vector=[(0,-100),(1,10),(2,100)]
    feature.error=2;
    feature.margin=0;
    feature.toggle=1;
    
    
    mytrainer.features=[feature];
    mytrainer.minimize_threshold_toggle(feature);
    
    print feature.threshold
    print feature.toggle;
    
    return;

def test_trainer4():
    pos_eg=2;
    neg_eg=2;
    mytrainer=trainer.Trainer();
    mytrainer.no_pos_training_eg=pos_eg;
    mytrainer.no_neg_training_eg=neg_eg;
    mysample=sample_generate.TestImages();
    mysample.gen_center_feature();
    mysample.show_image();
    
    #mysample.show_image();
    #implement get_training_examples();
    this_pos_example=example_images.Example(mysample.noisy_imgarr,True,float(1./pos_eg));
    mytrainer.pos_training_examples=[this_pos_example]*pos_eg;
    
    mysample.reset_image();
    mysample.gen_neg_blank();
    #mysample.show_image();
    this_neg_example=example_images.Example(mysample.noisy_imgarr,False,float(1./neg_eg));
    mytrainer.neg_training_examples=[this_neg_example]*neg_eg;

    mytrainer.training_examples=mytrainer.pos_training_examples;
    mytrainer.training_examples.extend(mytrainer.neg_training_examples);
    
    #print mytrainer.training_examples[2]
    
    mytrainer.training_rounds=1;
    
    mytrainer.generate_features();
    print "Features Generated....."
    mytrainer.init_adaboost();
    print "Feature Vectors Initialized...."
    mytrainer.adaboost_main_loop();
    
    [alpha,feature]=mytrainer.adaboost_rule[0];
    
    
    for rect in feature.rectlist:
        print rect;
    
    print len(mytrainer.features);
    print "threshold: "+str(feature.threshold);
    print "response: "+str(feature.response_vector);
    print "error: "+str(feature.error);
    print "toggle: "+str(feature.toggle);
    
    feature.show_feature();
    
    #log all features to logfile
    #for thisfeature in mytrainer.features:
    #    strmsg=' ';
    #    for rectval in thisfeature.rectlist:
    #        strmsg=strmsg+str(rectval)+'\n';
    #    strmsg=strmsg+str(thisfeature.response_vector);
    #    strmsg=strmsg+'\n';
    #    logging.debug(strmsg);
     
    return

                            
def main():
    test_trainer2();
    #test_trainer3();
    #test_trainer4();

if __name__ == "__main__":
    main()  