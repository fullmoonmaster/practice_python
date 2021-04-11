#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy
import scipy.special


# In[28]:


class neuralNetwork:
    
    #initialize
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        #each Node for layers
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        #set Learning Rate
        self.learningRate = learningRate
        
        self.weightInput = (numpy.random.normal(0.0,pow(self.hiddenNodes,-0.5)),(self.hiddenNodes,self.inputNodes))
        self.weightHidden = (numpy.random.normal(0.0,pow(self.outputNodes,-0.5)),(self.outputNodes,self.hiddenNodes))
    
        #define activation function(sigmoid)
        self.activation_function = lambda x:scipy.special.expit(x)
        
    
    #Training NeuralNetwork 
    def train():
        pass
    
    #Query
    def query(self, input_list):
        
        inputs = numpy.array(input_list,ndmin=2).T
        
        #Inputs of hidden nodes
        hidden_inputs = numpy.dot(self.weightInput,inputs)
        #outputs of hidden nodes
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #final inputs for last layers
        final_inputs = numpy.dot(self.weightHidden,hidden_outputs)
        #final output of last layers
        final_output = self.activation_function(final_inputs)
        
        return final_outputs


# In[34]:


input_Nodes = 3
hidden_Nodes = 3
output_Nodes = 3
learning_Rate = 0.3

n = neuralNetwork(input_Nodes,hidden_Nodes,output_Nodes,learning_Rate)


# In[43]:





# In[ ]:





# In[ ]:





# In[ ]:




