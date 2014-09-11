import image_generate
reload(image_generate)
from image_generate import GenerateImage


def main():
    snr=1.5;
    genimg=GenerateImage(snr);
    genimg.image_type='A';
    genimg.set_particle_shape('elongated');
    genimg.generate_img();
    genimg.save_image();
    genimg.lowres_image();
    genimg.save_image('A_lowres.jpg',highres=False);
    genimg.show_image();
    print "Saved Images"
    return;
if __name__ == "__main__":
    main()    