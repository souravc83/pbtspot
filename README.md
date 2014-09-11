pbtspot
=======

A python library which uses Probabilistic Boosting Tree for  Spot detection in fluoroscence microscopyy images.

Background
----------
Spot Detection in fluorosence microscopy images is complicated, due to low signal to noise ratios. 
Machine learning methods are known to perform better than traditional image processing techniques in these situations.
This project uses a probabilistic boosting tree, which is a decision tree. Each node of the decision tree is a strong 
Adaboost classifier (a linear combination of a set of weak Haar-like features). Such classifiers are used widely in face 
recognition, following the celebrated paper of Viola and Jones.
The main goal of this project was to improve upon the Viola-Jones classifier using a probabilistic method, which 
can then be used subsequently for Bayesian tracking methods, which have become popular in fluoroscence particle 
tracking in the recent past.

Installation
----------
    python setup.py build
    sudo python setup.py install 
    (On a windows system, simply use "python setup.py install")


    
