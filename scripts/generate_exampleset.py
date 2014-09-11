#global imports
#local imports
from pbtspot.adaboost import example_images


def generate_images(SNRval=0.5,pos_eg=4000,neg_eg=4000,egtype='Training'):
    eg_img=example_images.ExampleImages(SNRval,pos_eg,neg_eg);
    eg_img.get_new_examples();
    eg_img.save_examples(egtype);
    return;

def gen_exampleset():
    generate_images(0.5,4000,4000,'Training')
    print "Generated Training Examples....."
    generate_images(0.5,4000,4000,'Testing')
    print "Generated Testing Examples...."
    return;

if __name__=="__main__":
    gen_exampleset()
    
