import numpy as np
import NeuralNetwork as nn
import random


il = np.zeros((2,1))
hl1 = np.ones((2,1))
ol = np.ones((1,1))
layers = [il,hl1,ol]

network = nn.NeuralNetwork(layers)


inputs = np.array([1,0])
inputs.shape = (2,1)
outputs = np.array([1,0])
outputs.shape = (2,1)

data = [np.array([1,1]),np.array([1,0]),np.array([0,0]),np.array([0,1])]
answers = [np.zeros((1,1)),np.ones((1,1)),np.zeros((1,1)),np.ones((1,1))]
for i in data:
  i.shape = (2,1)
  
'''
for x in range(100000):
    index = random.randint(0,3)
    dataPoint = data[index]
    network.trainStochastically(dataPoint, answers[index])

'''
for x in range(100000):
    
    for i in range(len(data)):
        dataPoint = data[i]
        network.computeGradientsAndDeltas(dataPoint, answers[i])
    
    if(x%5==0):
        network.updateWeightsAndBiases()


for q in range(len(data)):
  network.layers[0] = data[q]
  print(network.feedForward())
  print(answers[q])

