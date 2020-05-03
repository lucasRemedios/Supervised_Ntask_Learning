# Supervised_Ntask_Learning 

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

# Desired behavior of experiment :
### 1) Dynamic testing 
### 2) Dynamic training
### 3) Dynamic testing
#
# Where it stands:
### Issues with the dynamic testing
### So the experiment currently is:
### 1) Dynamic training
### 2) STATIC testing (just see that the model has learned the tasks)
### 3) Dynamic testing (fails--something is wrong with when contexts are switched)

