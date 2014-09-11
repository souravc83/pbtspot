import pbtspot
from pbtspot import set_default,train_and_test, globalconstants
import matplotlib.pyplot as plt


set_default.set_default_constants();
#set constants
glob_const=globalconstants.GlobalConstants();
glob_const.no_pos_training_eg=50;
glob_const.no_neg_training_eg=50;
glob_const.no_pos_testing_eg=1000;
glob_const.no_neg_testing_eg=1000;
glob_const.savefile();

#main script
SNR=0.5;
train_test=train_and_test.TrainAndTest(SNR);
train_test.load_saved_tree();
#train_test.train_tree();
#train_test.save_tree();
train_test.test_tree();
[tpr,fpr]=train_test.get_roc_curve();
plt.plot(fpr,tpr,'ro-');
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05,1.05)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
print "tpr: "+str(tpr)
print "fpr: "+str(fpr)
plt.show();
