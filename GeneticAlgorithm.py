import numpy as np
import NeuralNetwork
import os
import math
import random

class GeneticAlgorithm:
    
    def __init__(self, layerStructure, dataFolderPath, baseBreedingProbability=1, mutationRate = 0.01):
        # Defines the layer structure through the constructor
        self.layerStructure = layerStructure
        # Initialises the value of the gene pool list
        self.genePool = []
        # Defaults to 1
        self.baseBreedingProbability = baseBreedingProbability
        # Defaults to 0.01
        self.mutationRate = mutationRate
        # If the path doesn't already exist
        if(os.path.exists(dataFolderPath)==False):
            # Create the path
            os.mkdir(dataFolderPath)
            
        # Stores the data folder path in an object variable    
        self.dataFolderPath = dataFolderPath
        # Defines the path of the folder for the weights to go into
        self.weightPath = self.dataFolderPath + "/weights"
        # Defines the path of the folder for the biases to go into
        self.biasPath = self.dataFolderPath + "/biases"
        if(os.path.exists(self.weightPath)==False):
            # Create the path
            os.mkdir(self.weightPath) 
        # If the path doesn't already exist
        if(os.path.exists(self.biasPath)==False):
            # Create the path
            os.mkdir(self.biasPath)
        # Initializes the total score variable which stores the value of all scores added together
        self.totalScore = 0
    
    def generateGenePool(self, instanceCount, weightFilePaths=None, biasFilePaths=None):
        # Resets gene pool list
        self.genePool = []
        # If files were specefied to load into this gene pool
        if(weightFilePaths==None or biasFilePaths==None):
            # Repeat as many times as specified by instanceCount
            for i in range(instanceCount):
                # Add a network instance with randomized weights and biases and index it by the current iteration
                self.genePool.append(NetworkInstance(NeuralNetwork.NeuralNetwork(self.layerStructure), i))
        # If the file path lists aren't the same length as the instance count
        elif(len(weightFilePaths) != instanceCount or len(biasFilePaths) != instanceCount):
            # Print warning
            print("File path lists must be the same length as the instanceCount")
        # If all is right and files were given
        else:
            # Repeat as many times as specified by instanceCount
            for i in range(instanceCount):
                # Add a network instance with specific weights and biases from a file and name it by the current iteration
                self.genePool.append(NetworkInstance(NeuralNetwork.NeuralNetwork(self.layerStructure, weightFilePaths[i], biasFilePaths[i]), i))
                
    def generateNextGenePool(self, instanceCount):
        # Update the total score variable
        self.getTotalScore()
        # Will be storing network instances, repeated based off their score
        weightedGenePool = []
        # Will contain the values that will replace the gene pool list
        newPool = []
        # Loop through every instance in the gene pool
        for i in self.genePool:
            # Repeat based off percent calculation
            for j in range(self.getScoreAsPercent(i.score)):
                # Add index of instance to the list (using index over object to be more efficient)
                weightedGenePool.append(i.index)
                
        # Repeat instanceCount times
        for x in range(instanceCount):
            # Get 2 random breeders based off of the index randomly generated from the weighted gene pool list
            breederA = self.genePool[weightedGenePool[random.randint(0, len(weightedGenePool)-1)]]
            breederB = self.genePool[weightedGenePool[random.randint(0, len(weightedGenePool)-1)]]
            # Adds the newly breeded instace to the newPool list
            newPool.append(self.crossoverGenes(breederA, breederB, x))
            
        # Sets the gene pool to a copy of the newPool list
        self.genePool = newPool.copy()
        
    def getTotalScore(self):
        # If there isn't a gene pool then return out of the function
        if(len(self.genePool)<1):
            return
        # Will represent the sum of the scores of the gene pool
        scoreSum = 0
        # Loop through the gene pool
        for i in self.genePool:
            # Add the score of that instance to the sum
            scoreSum += i.score
        # Update the variable
        self.totalScore = scoreSum
        # Also return the sum
        return scoreSum
    
    def getScoreAsPercent(self, score):
        # Get the ratio of this score to the total score as a percent
        percent = math.floor((score/self.totalScore)*100)
        # Don't allow the percent to fall below the base breeding probability
        if(percent < self.baseBreedingProbability):
            percent = self.baseBreedingProbability
        # Return the percent
        return percent
    
    def crossoverGenes(self, instanceA, instanceB, index):
        # This instance will be returned after it is "bred"
        returnInstance = NetworkInstance(NeuralNetwork.NeuralNetwork(self.layerStructure), index)
        # Loop through weight matrices
        for i in range(len(returnInstance.neuralNet.weightMatrices)):
            # Defines the iteration of this specific matrix
            iterator = np.nditer(returnInstance.neuralNet.weightMatrices[i], flags=["multi_index"])
            # Loops until iterator has iterated the entire matrix
            while not iterator.finished:
                # Generates random boolean
                randBool = bool(random.getrandbits(1))
                # Generates float from 0 to 1
                randFloat = random.uniform(0,1)
                # If the random number is less than the mutation rate
                if(randFloat<self.mutationRate):
                    # Set this weight to a random number
                    returnInstance.neuralNet.weightMatrices[i][iterator.multi_index] = random.uniform(returnInstance.neuralNet.weightConstant, -returnInstance.neuralNet.weightConstant)   
                # If the random boolean was true
                elif(randBool):
                    # The value at this index in this matrix is equal to that value in instance a
                    returnInstance.neuralNet.weightMatrices[i][iterator.multi_index] = instanceA.neuralNet.weightMatrices[i][iterator.multi_index]
                # If the random boolean was false
                else:
                    returnInstance.neuralNet.weightMatrices[i][iterator.multi_index] = instanceB.neuralNet.weightMatrices[i][iterator.multi_index]
                # Move on to the next iteration
                iterator.iternext()
        # Returns the instance
        return returnInstance
        
        # Loop through bias matrices
        for i in range(len(returnInstance.neuralNet.biasMatrices)):
            # Defines the iteration of this specific matrix
            iterator = np.nditer(returnInstance.neuralNet.biasMatrices[i], flags=["multi_index"])
            # Loops until iterator has iterated the entire matrix
            while not iterator.finished:
                # Generates random boolean
                randBool = bool(random.getrandbits(1))
                # Generates float from 0 to 1
                randFloat = random.uniform(0,1)
                # If the random number is less than the mutation rate
                if(randFloat<self.mutationRate):
                    # Set this bias to a random number
                    returnInstance.neuralNet.weightMatrices[i][iterator.multi_index] = random.uniform(returnInstance.neuralNet.biasConstant, -returnInstance.neuralNet.biasConstant)     
                # If the random boolean was true
                elif(randBool):
                    # The value at this index in this matrix is equal to that value in instance a
                    returnInstance.neuralNet.biasMatrices[i][iterator.multi_index] = instanceA.neuralNet.biasMatrices[i][iterator.multi_index]
                # If the random boolean was false
                else:
                    returnInstance.neuralNet.biasMatrices[i][iterator.multi_index] = instanceB.neuralNet.biasMatrices[i][iterator.multi_index]
                # Move on to the next iteration
                iterator.iternext()
                                  
        # Returns the instance
        return returnInstance
    
    def saveGenePool(self):
        # This makes sure the program doesn't error even if the user deletes any folders
        # If the path doesn't already exist
        if(os.path.exists(self.dataFolderPath)==False):
            # Create the path
            os.mkdir(self.dataFolderPath)        
        # If the path doesn't already exist
        if(os.path.exists(self.weightPath)==False):
            # Create the path
            os.mkdir(self.weightPath) 
        # If the path doesn't already exist
        if(os.path.exists(self.biasPath)==False):
            # Create the path
            os.mkdir(self.biasPath)
        
        for i in self.genePool:
            # Saves the weights of i as a text file in the weights path named after its index
            # EXAMPLE: 23w.txt for the 23rd instance in the gene pool
            i.neuralNet.saveWeightMatrices(self.weightPath + "/" + str(i.index) + "w.txt")
            # Does the same to the biases except it adds the b suffix instead of the w suffix
            i.neuralNet.saveBiasMatrices(self.biasPath + "/" + str(i.index) + "b.txt")
    
    @staticmethod
    def getFileListFromDirectory(directory):
        # Gets all file names from the directory
        fileList = os.listdir(directory)
        # Loops through fileList
        for i in range(len(fileList)):
            # Adds the beginning part of the directory to the file name
            fileList[i] = directory + "/" + fileList[i]
            
        # Returns the list with full file paths
        return fileList
    
    
class NetworkInstance:
    
    def __init__(self, neuralNet, index):
        #Initializes variables
        self.neuralNet= neuralNet
        self.index = index
        self.score = 0
        
'''       
il = np.array([(4,10,3)])
il.shape = (3,1)
hl1 = np.ones((4,1))
hl2 = np.ones((3,1))
ol = np.ones((2,1))

layers = [il,hl1,hl2,ol]

ga = GeneticAlgorithm(layers, os.getcwd()+"/files")
ga.generateGenePool(10)
ga.saveGenePool()

print(ga.genePool[2].neuralNet.weightMatrices)

ga.generateGenePool(10, GeneticAlgorithm.getFileListFromDirectory(ga.weightPath),GeneticAlgorithm.getFileListFromDirectory(ga.biasPath))
print("\n")
print(ga.genePool[2].neuralNet.weightMatrices)
'''