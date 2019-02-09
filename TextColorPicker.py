import NeuralNetwork as nn
import GeneticAlgorithm as ga
import numpy as np
import tkinter as tk
import random
import os
import math

# Input layer structure
il = np.array([(0,0,0,0,0,0)])
il.shape = (6,1)
# Hidden layer structure
hl1 = np.zeros((4,1))
# Output layer structure
ol = np.zeros((2,1))
# The complete layer structure
layers = [il,hl1,ol]

geneticAlgorithm = ga.GeneticAlgorithm(layers, os.getcwd()+"/DNA")

window = tk.Tk()
window.title("Text Color Picker")

def rgb(inputRgb):
    return "#%02x%02x%02x" % inputRgb   


genePoolSize = 50
repeatSize = 25

# Creates random gene pool of 15
geneticAlgorithm.generateGenePool(genePoolSize, weightFilePaths=ga.GeneticAlgorithm.getFileListFromDirectory(geneticAlgorithm.weightPath), biasFilePaths=ga.GeneticAlgorithm.getFileListFromDirectory(geneticAlgorithm.biasPath))

# Stores the iteration of instance list we are on
currentInstanceIteration = 0
# The generation we are on
currentGeneration = 1

# The labels with white text over the random background
label1 = tk.Label(window, bg="Black", fg=rgb((255,255,255)), width=30, height=10, text="hello world", font="none 18 bold")
label1.grid(row=0, column=0)
# The label with black text over the random background
label2 = tk.Label(window, bg="Black", fg=rgb((0,0,0)), width=30, height=10, text="hello world", font="none 18 bold")
label2.grid(row=0, column=1)

# Stores all the labels representing the responses to the left option
leftCpuResponseLabels = []
# Stores all the labels representing the responses to the right option
rightCpuResponseLabels = []
# Stores all the labels representing the score of each instance
scoreLabels = []

# Stores confidence in left option
leftConfidenceLabel = tk.Label(window, bg="Black", fg=rgb((255,255,255)))
leftConfidenceLabel.grid(row=2, column=0)

# Stores confidence in right option
rightConfidenceLabel = tk.Label(window, bg="Black", fg=rgb((255,255,255)))
rightConfidenceLabel.grid(row=2, column=1)

for i in range(len(geneticAlgorithm.genePool)):
    # Populates both lists
    leftCpuResponseLabels.append(tk.Label(window, bg=rgb((0,0,0))))
    leftCpuResponseLabels[i].grid(row=3+i, column=0)
    
    rightCpuResponseLabels.append(tk.Label(window, bg=rgb((0,0,0))))
    rightCpuResponseLabels[i].grid(row=3+i, column=1)
    
    scoreLabels.append(tk.Label(window, bg=rgb((0,0,0)), fg="White"))
    scoreLabels[i].grid(row=3+i, column=2)

generationLabel = tk.Label(window, bg="Black", fg="White", font="none 14 bold", text="Generation: " + str(currentGeneration))
generationLabel.grid(row=1, column=2)

def chooseNewColor():
    # Randomly generates r g and b values
    r=random.randint(0, 255)
    g=random.randint(0, 255)
    b=random.randint(0, 255)    
    # Stores as a variable
    randomColor = rgb((r,g,b))
    # Gets the random color as a numpy array
    inputArray = np.array([(r/255,g/255,b/255,r/255,g/255,b/255)])
    inputArray.shape= (6,1)
    
    # Loops through every instance in the gene pool    
    for i in range(len(geneticAlgorithm.genePool)):
        # Sets the input of the current instance to be this new color
        geneticAlgorithm.genePool[i].neuralNet.layers[0] = inputArray
        # Preforms a feed forward on the current instance's neural net
        geneticAlgorithm.genePool[i].neuralNet.feedForward()

    # Displays which option the computer predicts is correct
    displayComputerGuess()
    # Sets the background colors of the labels to the new random color
    label1.config(bg=randomColor)
    label2.config(bg=randomColor)
    # Updates confidene labels
    updateConfidenceLabels()
    # Update window
    window.mainloop()
    # Returns the color
    return randomColor

def determineScoreIncrease(userChoice, cpuPrediction):
    # If the index of the maximum value is equal on the prediction and the correct answer
    if(userChoice == np.argmax(cpuPrediction)):
        # The maximum score
        score = 10
        # The inverse confidence of the computer, multiplied by 10 and rounded
        difference = math.floor((1-cpuPrediction[np.argmax(cpuPrediction)])*10)
        # The score will be damaged less, the more the computer is confident in it's answer
        score-=difference
        return score
    # If not
    else:
        # Computer gets nothing
        return 0

def displayCpuScores():
    # Loop through the gene pool
    for i in range(len(geneticAlgorithm.genePool)):
        # Display corresponding score
        scoreLabels[i].config(text=geneticAlgorithm.genePool[i].score)

def updateConfidenceLabels():
    # Storing sum of confidence percent in left and right options
    lsum = 0
    rsum = 0
    for i in range(len(geneticAlgorithm.genePool)):
        # Caches outputlayer
        outputLayer = geneticAlgorithm.genePool[i].neuralNet.layers[len(geneticAlgorithm.genePool[i].neuralNet.layers)-1]
        # Adds percent confidence in each option to the left and right sums
        lsum += outputLayer[0]/np.sum(outputLayer)
        rsum += outputLayer[1]/np.sum(outputLayer)
    
    # Averages the sums
    lsum /= len(geneticAlgorithm.genePool)
    rsum /= len(geneticAlgorithm.genePool)
    
    # Updates labels
    leftConfidenceLabel.config(text=str(lsum))
    rightConfidenceLabel.config(text=str(rsum))
        
def leftClick():
    click(0)


def rightClick():
    click(1)

def click(inputArray):
    # References the global variable
    global currentInstanceIteration
    global currentGeneration
    
    userInput = inputArray
    
    # Loops through every instance in the gene pool    
    for i in range(len(geneticAlgorithm.genePool)):
        # Determines score increase based off of user input and output layer
        geneticAlgorithm.genePool[i].score += determineScoreIncrease(userInput, geneticAlgorithm.genePool[i].neuralNet.layers[len(geneticAlgorithm.genePool[i].neuralNet.layers)-1])
    # If this is the last test of the generation
    if((currentInstanceIteration+1)>repeatSize):
        # Generate the next gene pool
        geneticAlgorithm.generateNextGenePool(genePoolSize)
        # Reset the iteration
        currentInstanceIteration= 0
        # Increase the generation
        currentGeneration += 1
        # Update the generation label
        generationLabel.config(text="Generation: " + str(currentGeneration))
        # Save the gene pool
        geneticAlgorithm.saveGenePool()
    else:
        # Moves onto the next instance
        currentInstanceIteration +=1
    # Updates the score labels
    displayCpuScores()
    # Chooses next color
    chooseNewColor()

def displayComputerGuess():
    # Loops through every instance in the gene pool    
    for i in range(len(geneticAlgorithm.genePool)):
        # Caches outputlayer
        outputLayer = geneticAlgorithm.genePool[i].neuralNet.layers[len(geneticAlgorithm.genePool[i].neuralNet.layers)-1]
        # Stores the guess value as the index of the greatest value in the output layer  
        guess = np.argmax(outputLayer)
    
        if(guess==0):
            # Left label is green, right label is red
            leftCpuResponseLabels[i].config(bg=rgb((0,255,0)))
            rightCpuResponseLabels[i].config(bg=rgb((255,0,0)))
        else:
            # Right label is green, left label is red
            rightCpuResponseLabels[i].config(bg=rgb((0,255,0)))
            leftCpuResponseLabels[i].config(bg=rgb((255,0,0))) 
        
        # Displays the confidence as text
        leftCpuResponseLabels[i].config(text=outputLayer[0])
        rightCpuResponseLabels[i].config(text=outputLayer[1])

tk.Button(window, text="Choose", width=30, height=10, command=leftClick).grid(row=1, column=0)
tk.Button(window, text="Choose", width=30, height=10, command=rightClick).grid(row=1, column=1)

chooseNewColor()



