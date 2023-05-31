
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import random
from tqdm import tqdm
import pickle

class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    CENTER = 4


class TipCell:
    def __init__(self, i, j):
        self.x = j
        self.y = i
        self.life = 0
    
    def move(self, direction, width, height):
        if(direction == Direction.UP and self.y - 1 >= 1):
            self.y = self.y - 1
        elif(direction == Direction.DOWN and self.y + 1<= height - 2):
            self.y = self.y + 1
        elif(direction == Direction.LEFT and self.x - 1 >= 1):
            self.x = self.x - 1
        elif(direction == Direction.RIGHT and self.x + 1 <= width - 2):
            self.x = self.x + 1
        
        self.life = self.life + 1
        
        


class AngiogenesisModel:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.k = 1/(((2*10**-3)**2)/(2.9*10**-7))
        self.h = 0.005
        self.D = 0.00035
        self.alpha = 0.6
        self.chi0 = 0.38
        self.rho = 0.34
        #self.rho = 0
        self.beta = 0.05
        self.gamma = 0.1
        self.eta = 0.1
        self.tAge = 2
        self.delta = 1
        self.kn = 0.75
        self.maxBranches = 50
        
        self.resetSystem()
        
        
    def resetSystem(self):
        #Endothelial cells
        self.nInitial = np.zeros((self.height, self.width))
        #Fibronectin
        self.fInitial = np.zeros((self.height, self.width))
        #angiogenic factors 
        self.cInitial = np.zeros((self.height, self.width))
        
        self.tipCells = []
        
        self.occupiedCells = np.zeros((self.height, self.width))
    
    def resetOccupiedCells(self):
        self.occupiedCells = np.zeros((self.height, self.width))
    
    def setInitialConcentrations(self, prolifLocations):
        nProlif = sum(sum(prolifLocations))

        for i in range(1,self.height-1):
            for j in range(1,self.width-1):
                if(prolifLocations[i,j] == 1):
                    self.cInitial[i,j] = 0.5
                    for s in range(1,self.height-1):
                        for l in range(1,self.width-1):
                            distance = np.sqrt(((s - i)*self.h)**2 + ((l - j)*self.h)**2)
                            if(distance < 0.1):
                                self.cInitial[s,l] = self.cInitial[s,l] + 2
                            else:
                                nu = (np.sqrt(5) - 0.1)/(np.sqrt(5) - 1)
                                self.cInitial[s,l] = self.cInitial[s,l] + (((nu - distance)**2)/(nu - 0.1))
                            
                            #else:
                                #self.cInitial[s,l] = self.cInitial[s,l] + 1
                    
                self.fInitial[i,j] = 0.4
                self.nInitial[i,j] = 0
        self.cInitial = self.cInitial/(np.max(self.cInitial))
                
                
    def setTipCells(self, tipCellLocations):
        n = np.size(tipCellLocations,0)
        m = np.size(tipCellLocations,1)
        if(n != self.height or m != self.width):
            print("Matrix sizes don't coincide")
            return -1
        
        for i in range(0,n):
            for j in range(0,m):
                if(tipCellLocations[i,j] == 1):
                    self.tipCells.append(TipCell(i,j))
                    self.occupiedCells[i,j] = 1
                    self.nInitial[i,j] = 1
                    #for s in range(1,self.height-1):
                        #for l in range(1, self.width - 1):
                            #if(i != s or j != l):
                                #distance = np.sqrt((s - i)**2 + (l - j)**2)
                                #self.nInitial[s,l] = self.nInitial[s,l] + (1/distance)
                            #else:
                                #self.nInitial[s,l] = self.nInitial[s,l] + 1
                    
        #self.nInitial = self.nInitial/(sum(sum(self.nInitial)))
                    
                
                
    
    
    def evolveSystem(self, nSteps):
        self.nMatrix = np.zeros((self.height, self.width, nSteps + 1))
        self.cMatrix = np.zeros((self.height, self.width, nSteps + 1))
        self.fMatrix = np.zeros((self.height, self.width, nSteps + 1))
        self.occupiedCellsMovie = np.zeros((self.height, self.width, nSteps + 1))
        
        self.nMatrix[:,:,0] = self.nInitial
        self.cMatrix[:,:,0] = self.cInitial
        self.fMatrix[:,:,0] = self.fInitial
        self.occupiedCellsMovie[:,:,0] = self.occupiedCells
        
        for i in tqdm(range(1,nSteps+1)):
            self.step(i)
            self.occupiedCellsMovie[:,:,i] = self.occupiedCells
            
    def step(self, step):
        c = self.cMatrix[:,:,step-1]
        f = self.fMatrix[:,:,step-1]
        n = self.nMatrix[:,:,step-1]
        for i in range(1,self.height-1):
            for j in range(1,self.width-1):
                p0 = 1 - ((4*self.k*self.D/(self.h**2))) + ((self.k*self.alpha*self.chi(c[i,j]))/(4*(self.h**2)*(1 + self.alpha*c[i,j])))*((c[i, j+1] - c[i, j-1])**2 + (c[i-1,j] - c[i+1,j])**2)-((self.k*self.chi(c[i,j]))/(self.h**2))*(c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1] -4*c[i,j]) - ((self.k*self.rho)/(self.h**2))*(f[i+1, j] + f[i-1, j] - 4*f[i,j] + f[i,j+1] + f[i,j-1])
                p1 = ((self.k*self.D)/(self.h**2)) - (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i,j+1] - c[i,j-1]) + self.rho*(f[i,j+1] - f[i, j-1]))
                p2 = ((self.k*self.D)/(self.h**2)) + (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i,j+1] - c[i,j-1]) + self.rho*(f[i,j+1] - f[i, j-1]))
                p3 = ((self.k*self.D)/(self.h**2)) + (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i-1,j] - c[i+1,j]) + self.rho*(f[i-1,j] - f[i+1, j]))
                p4 = ((self.k*self.D)/(self.h**2)) - (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i-1,j] - c[i+1,j]) + self.rho*(f[i-1,j] - f[i+1, j]))
                    
                    
                
                probArray = self.getProbabilities([p0, p1, p2, p3, p4])
                p0 = probArray[0]
                p1 = probArray[1]
                p2 = probArray[2]
                p3 = probArray[3]
                p4 = probArray[4]
                
                
                self.nMatrix[i,j,step] = p0*self.occupiedCells[i,j] + p1*self.occupiedCells[i,j-1] + p2*self.occupiedCells[i,j+1] + p3*self.occupiedCells[i-1,j] + p4*self.occupiedCells[i+1,j]
                self.fMatrix[i,j,step] = f[i,j]*(1 - self.k*self.gamma*n[i,j]) + self.k*self.beta*n[i,j]
                self.cMatrix[i,j,step] = c[i,j]*(1 - self.k*self.eta*n[i,j])
                
                if(self.tipOccupied(i,j)):
                    #print("Centro: "+str(p0))
                    #print("Izquierda: "+str(p1))
                    #print("Derecha: "+str(p2))
                    #print("Arriba: "+str(p3))
                    #print("Abajo: "+str(p4))
                    direction = self.getDirection(probArray)
                    index = self.getTipIndex(i,j)
                    self.tipCells[index].move(direction, self.width, self.height)
                    self.occupiedCells[self.tipCells[index].y, self.tipCells[index].x] = 1
                    oldTip = self.tipCells[index]
                    self.removeRepeatedTips(self.tipCells[index], index)
                    #Branching
                    availableBranchingPositions = self.getAvailableBranchingPositions(i,j)
                    index = self.getTipIndex(i,j)
                    if(oldTip.life > self.tAge and len(availableBranchingPositions) > 0 and n[i,j] > self.kn/c[i,j]):
                        Pn = c[i,j]/np.max(c)
                        if(random.random() < Pn and len(self.tipCells) < self.maxBranches):
                            positionIndex = random.randint(0,len(availableBranchingPositions)- 1)
                            branchPosition = availableBranchingPositions[positionIndex]
                            newTip = TipCell(branchPosition[0], branchPosition[1])
                            self.tipCells.append(newTip)
                            self.occupiedCells[newTip.y, newTip.x] = 1
                            self.removeRepeatedTips(newTip, len(self.tipCells)-1)
                        
                            
                    
                    
    
    def getAvailableBranchingPositions(self, i,j):
        availablePositions = []
        for s in range(-1,2):
            for l in range(-1,2):
                if(not self.tipOccupied(i + s, j + l)):
                    availablePositions.append([i+s, j+l])
        
        return availablePositions
    
    def chi(self, x):
        return self.chi0/(1 + self.delta*x)
    
    def tipOccupied(self, i,j):
        for s in range(0,len(self.tipCells)):
            if(self.tipCells[s].x == j and self.tipCells[s].y == i):
                return True
        
        return False
    
    def getTipIndex(self, i,j):
        for s in range(0,len(self.tipCells)):
            if(self.tipCells[s].x == j and self.tipCells[s].y == i):
                return s
    def getProbabilities(self, probArray):
        
        
        probArray = np.array(probArray)
        maxVal = np.max(probArray)
        minVal = np.min(probArray)
        for i in range(0,len(probArray)):
            probArray[i] = (probArray[i] - minVal)/(maxVal - minVal)
        
        probArray = probArray/sum(probArray)
        return probArray
    
    def getDirection(self, probArray):
    
        cumulative = 0
        rand = random.random()
        for i in range(0,len(probArray)):
            if(rand >= cumulative and rand <= cumulative + probArray[i]):
                
                if(i == 0):
                    return Direction.CENTER
                elif(i == 1):
                    return Direction.LEFT
                elif(i == 2):
                    return Direction.RIGHT
                elif(i == 3):
                    return Direction.UP
                elif(i == 4):
                    return Direction.DOWN
            else:
                cumulative = cumulative + probArray[i]
    
    def removeRepeatedTips(self, tip, index):
        tipsToRemove = []
        for i in range(0, len(self.tipCells)):
            if(index != i and tip.x == self.tipCells[i].x and tip.y == self.tipCells[i].y):
                tipsToRemove.append(self.tipCells[i])
                
        for i in range(0,len(tipsToRemove)):
            self.tipCells.remove(tipsToRemove[i])
        #if(len(indexToRemove) > 0):
            #print("Removed "+str(len(indexToRemove))+ " Branches")
        
    
    def getPicture(self, tipCellLocations, proliferatingLocations, vasculaturePicture):
        
        tipCellColor = [224/255, 190/255, 79/255]
        proliferatingColor = [58/255.0, 252/255.0, 84/255.0]
        vasculatureColor = [201/255, 74/255, 0/255]
        backgroundColor = [1, 1, 1]
        
        picture = np.ones((self.height, self.width, 3))
        
        for i in range(0,self.height):
            for j in range(0,self.width):
                if(vasculaturePicture[i,j] == 1):
                    picture[i,j,:] = vasculatureColor
            
                if(proliferatingLocations[i,j] == 1):
                    picture[i,j,:] = proliferatingColor
                       
                if(tipCellLocations[i,j] == 1):
                    picture[i,j,:] = tipCellColor
                    
                if(proliferatingLocations[i,j] == 0 and tipCellLocations[i,j] == 0 and vasculaturePicture[i,j] == 0):
                    picture[i,j,:] = backgroundColor
        
        return picture
    
    
    def saveBloodVesselNetworkInstance(self, fileName):
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
            print("Object saved to "+fileName)
    
    
    def openBloodVesselNetworkInstance(self, fileName):
        bloodNetworkInstance = pickle.load(open(fileName, 'rb'))
        return bloodNetworkInstance

