"""
This script plots training and testing accuracies as a function of different variables
"""
import train_and_test
reload(train_and_test)
import matplotlib.pyplot as plt

def precisionrecallscript():
    mytraintest=train_and_test.TrainAndTest(0.5);
    
    #set constants

    mytraintest.train_tree();
    mytraintest.test_tree_whole_image();

    precision=mytraintest.precision;
    recall=mytraintest.recall;
    
    print precision
    print recall
    
    plt.plot(recall,precision,'ro')
    plt.show()
    
    return;
    
if __name__ == "__main__":
    precisionrecallscript()
