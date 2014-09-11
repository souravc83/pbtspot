#input 
#write_cascade_info()


#functions
#class Curr_Image
#calc_integral_image() What class to place this in?

#Base class: features
#derived class: type of features??

#class myCascade
#class which gives no of stages and classifiers at each stage
#must read from a xml.




#training

#class trainer
#variables: weights
#variables: response matrix
#var: no of classifiers
#should we take out classifiers previously used? need to store them in that case.


#writer:writes alpha info for each cascade
#different files for each cascade? is there a json type file format?

#input image? object? locations of object?

#questions
#how to make mean of image 0 and variance 1??
#how to handle case when many features have same value: line 27-32 of algorithm 4


#calculate_weights() #todo
#initialize and normalize weights
#algorithm 4: for each feature find
#threshold,toggle,error,margin

#Calculate integral image.

#calculate_response_matrix()
#for each Haar feature, calculate estimate
#calculate Haar feature response matrix no. of features * no of training eg.

#calculate threshold value for each feature from the response matrix;

#there's some stuff that needs to be done only for the first cascade
#fetch_training_examples()#both positive and negative training examples
#calculate_response_matrix()



#for each cascade:

#for no. of features in present cascade:
    #calculate best threshold for each feature.
    #select best feature
    #update weights of examples
#calculate final alphas
#return sum(alpha_t*h_t)

#save cascade info
#save in xml?


#detection
# input image sliding window
#for each cascade
    #test on cascade
    #reject if fails on cascade
    #break