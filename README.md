# Supervised_Ntask_Learning 
#
# Motivation:
#### Humans can perform multiple tasks on the fly. When an approach doesn't succeed, the human can switch its approach, or context that it is looking at the task with, on the fly, to attempt to successfully complete the task. We want to match this type of behavior with a neural network. A model should be able to switch its context on the fly (dynamically) while training and testing, where tasks are randomly cycled through and given to the model.


# There are 2 significant files:
# 1. Context_Layer.py : 
  #### Contains the Ntask Keras layer
#
# 2. logic_gate_experiment.ipynb : 
  #### Contains the supervised Ntask learning experiment on learning 8 logic gates mapped to 8 contexts (using the Context Layer) in 1 neural network
#
# Instructions:
#
### Install requirements from requirements.txt
#
### Run logic_gate_experiment.ipynb. -- issues with experiment are noted in this file

#

# A. Desired behavior of experiment :
### 1) Dynamic testing 
### 2) Dynamic training
### 3) Dynamic testing
#
# B. Where it stands currently: 
### Issues with the dynamic testing
### So the experiment currently is:
### 1) Dynamic training
### 2) STATIC testing (just see that the model has learned the tasks)
### 3) Dynamic testing (fails--something is wrong with when contexts are switched)

#
# Once the logic gate experiment can perform the steps listed in section A. 
#### the experiment should be repeated with n mnist task variants (ex: is it odd, is it even, div by 3, div by 5, etc)
# The model is capable of learning these mappings as seen in the experiment in the MNIST folder, but this is not dynamic training -- the model is forced to learn task 0 on context 0, task 1 on context 1, etc.

# 
# Examples of desired graphs at the end of these experiments can be seen in the Graphs folder

#
# Picture / Math to help understand training algorithm ie the train() function in logic_gate_experiment.ipynb can be found in Algo folder.


#
# # *** Where might errors lie?
- maybe in the diff_errs and moving_avg_context_loss
    - this is a kind of thresholding system, which may work for training but not work for testing
    - potentially, with the existing code, if the model was trained better, it could be overfit to each of the 8 tasks which would hopefully allow it to switch faster and stay on the correct context
    
- maybe an issue with sharing or not sharing global variables between training and testing (ex: diff_errs, moving_avg_context_loss, etc.)
    - these maybe could be thought of as "learned weights" and are getting shared incorrectly or reset etc before testing.
    
- maybe the model needs to be trained better

- maybe there is a better way to trigger task switching and perseverating
   
- maybe testing was implemented incorrectly

