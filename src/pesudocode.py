

generate_training_eg(no_examples):
	img=[];
	for i in range(no_examples):
		img.append(Image_generate.generate_image());

	return img;

def main():
	train_imgs=generate_training_eg(100000);#info about spot locations also needed
	#should we pass images one by one?
	#positive and negative eg ratio?
	
	classifier=viola_jones();
	classifier.train(train_imgs);
