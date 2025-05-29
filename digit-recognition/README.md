# Digit Recognition  
The code in this directory is inspired by one of the mandatory assignments in module 2 of the
[Machine Learning Specialization course](https://www.coursera.org/specializations/machine-learning-introduction)
By Andrew Ng at Coursera/Stanford.

## Quick start
 ``` python3 main.py```

This command will both train the model, make inference, and compare results from training data with the results from validation data.

I assume here that you have all the necessary libraries installed (will be explained later)

Please play around with the parameters in  [config.py](config.py)

## Introduction
The original assignment uses deep neural networks to distinguish ZERO and ONES using a dense neural network.

I have done the following modifications and improvements.

* Rewritten the code to run as a regular python code instead of Jupyter notebook.
* Refactored and simplified the original code
* Implemented the support for identifying all digits 0 - 9 using one shot encoding. That is multi class classification, not just binary classification.
* Split the data into a training set and validation set to assess how the network performs on unseen data
* The training set and the validation set is chooses randomly (which means that the results might vary slightly between each time you run the code) 
* Made the following parameters configurable via dedicated config fil [config.py](config.py)
    * The number of EPOCS to run
    * The NN layers to use
    * The fraction of samples to use for training and validation
    * The parameter used for the Adam Optimization (typically somewhere around 0.001)
* Printing out and compare the performance of the network on the training data vs validation data (a huge difference means over fitting)
* Plotting of miss classified data.

## Data Set
The data set is the set provided by the Coursera course (Its not the MINST set). The image resolution is only 20x20 pixels which poses some challenges.   

## Requirements
### Packages 
TBD
### Cuda
You should have CUDA installed (not covered here, use Google). The code has only been tested with a NVIDIA screen card. Its very important that you utilize the capabilities of your screen card, otherwise the code will run very slow.

## Performance
For most parameter setting the code should never take more than a couple of minutes to run if you utilize your screen card (GPU).

In terms of accuracy the best result I have been able to achieve so far is 0% error on the training set, and 5.5% error on the validation set. This could point towards and over fitting problem, however many of the misidentified images in the validation set is very difficult to identify even with the human eye.

This result was achieved with the following settings (see [config.py](config.py) ).

* ` NO_EPOCS = 200  # Number of training passes`

* `LAYERS = [512, 512, 10] #Number  of nodes in the various neural network layers`

* `TEST_SIZE_FRACTION = 0.20 # The fraction of the data used for validation, must be in the range [0,1]`

* `ADAM = 0.001 gradient for the ADAM optimization`

I have also tried tuning various other hyper parameters (see code), without success, but Im working on it.

You can display miss identified images by setting 
`plot_misclassified=True` 
When calling
`print_statistics(......)` (see [main.py](main.py)) for examples.

## Todo
* Using a convolutional NN network would probably give better results. Also, using the MINST data set would probably yield better results since the resolution is higher and there is also a bigger number of training samples so that there will be less problems with over fitting. This will be the next steps.
* Make the various parameters configurable from the command line.

