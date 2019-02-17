import NeuralNetwork
import GeneticAlgorithm
import numpy as np
import os

genePoolSize = 100


il = np.zeros((18,1))
hl1 = np.ones((16,1))
hl2 = np.ones((16,1))
hl3 = np.ones((16,1))
hl4 = np.ones((16,1))
ol = np.ones((9,1))
layers = [il,hl1,hl2,ol]
ga = GeneticAlgorithm.GeneticAlgorithm(layers, os.getcwd()+"/TicTacToe1")
#ga.generateGenePool(genePoolSize)
#ga.saveGenePool()
ga.generateGenePool(genePoolSize,weightFilePaths=GeneticAlgorithm.GeneticAlgorithm.getFileListFromDirectory(ga.weightPath), biasFilePaths=GeneticAlgorithm.GeneticAlgorithm.getFileListFromDirectory(ga.biasPath))


ga2 = GeneticAlgorithm.GeneticAlgorithm(layers, os.getcwd()+"/TicTacToe2")
ga2.generateGenePool(genePoolSize, weightFilePaths=GeneticAlgorithm.GeneticAlgorithm.getFileListFromDirectory(ga2.weightPath), biasFilePaths=GeneticAlgorithm.GeneticAlgorithm.getFileListFromDirectory(ga2.biasPath))
#ga2.generateGenePool(genePoolSize)
#ga2.saveGenePool()
currentGeneIndex = 0

board = ["_"]*9
gameOver = False
winingTeam = ""


def move(index, letter):
  global gameOver
    
  board[int(index)] = letter

  print(board[0]+" "+board[1]+" "+board[2])
  print("\n")  
  print(board[3]+" "+board[4]+" "+board[5])
  print("\n")
  print(board[6]+" "+board[7]+" "+board[8])
  print("\n")


def convertBoardToArray():
  returnArray = np.zeros((len(board)*2,1))
  z = 0
  for i in range(len(board)):
    if(board[i] == "x"):
      returnArray[z] = 1
    elif(board[i] == "o"):
      returnArray[z+1] = 1
    z+=2
  return returnArray

def moveComputer(inputs, computer, team):
  moveValid = False
  # Set input layer
  computer.genePool[currentGeneIndex].neuralNet.layers[0] = inputs
  # Get high value moves in order
  outputs = computer.genePool[currentGeneIndex].neuralNet.feedForward()
  outputs.shape = (1,outputs.size)
  computerMoves = np.argsort(outputs)
  #print(ga.genePool[currentGeneIndex].neuralNet.feedForward())
  #print(computerMoves)
  sortIndex = 0
  while(moveValid==False):
    if(board[computerMoves[0,computerMoves.size-sortIndex-1]] == "_"):
      moveValid = True
    else:
      sortIndex+=1


  move(computerMoves[0,computerMoves.size-sortIndex-1], team)


def checkForWin():
    global winningTeam
    
    # Check for win
    for j in range(3):
        # Check each row
        if(checkRow(j)):
            return True
        # Check each column
        if(checkColumn(j)):
            return True
    
    # Check for tie
    boardFull = True
    
    for i in board:
        # If there is an empty space
        if(i=="_"):
            boardFull = False
            
    if(boardFull):
        # Tie
        winningTeam = "n"
        # Game has ended
        return True
    else:
        # Game has not ended
        return False
        

def checkRow(index):
    global winningTeam
    
    team = board[index*3]
    if(team == "_"):
        return False
    
    if(board[index*3+1] == team and board[index*3+2] == team):
        winningTeam = team
        return True

def checkColumn(index):
    global winningTeam
    
    team = board[index]
    if(team == "_"):
        return False
    
    if(board[index+3] == team and board[index+6] == team):
        winningTeam = team
        return True

'''
for i in range(200000):   
    gameOver = False
    while(not gameOver):
      inputs = convertBoardToArray()
      moveComputer(inputs, ga, "o")
      if(checkForWin()):
          break
      moveComputer(inputs, ga2, "x")
      if(checkForWin()):
          break
      gameOver = checkForWin()
    
    if(winningTeam=="x"):
        ga2.genePool[currentGeneIndex].score = 25
    if(winningTeam=="o"):
        ga.genePool[currentGeneIndex].score = 25
    else:
        ga.genePool[currentGeneIndex].score = 5
        ga2.genePool[currentGeneIndex].score = 5
    
    if(currentGeneIndex == len(ga.genePool)-1):
        ga.generateNextGenePool(genePoolSize)
        ga2.generateNextGenePool(genePoolSize)
        currentGeneIndex = 0
    else:
        currentGeneIndex+=1
        
    board = ["_"]*9
    winningTeam=""
ga.saveGenePool()
ga2.saveGenePool()   

'''

for i in range(100):

    while(not gameOver):
          inputs = convertBoardToArray()
          playerInput = input("Choose your move: 0-8")
          move(playerInput, "o")
          if(checkForWin()):
              break
          moveComputer(inputs, ga, "x")
          if(checkForWin()):
              break
          gameOver = checkForWin()
    
    print(winningTeam + " won!")
    board = ["_"]*9
    winningTeam=""
    currentGeneIndex+=1
