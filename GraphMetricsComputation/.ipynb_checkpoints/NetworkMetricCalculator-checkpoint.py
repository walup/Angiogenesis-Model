from enum import Enum
import numpy as np
import csv
from tqdm import tqdm
from IPython.display import clear_output
import random 
import matplotlib.pyplot as plt

class NetworkMetricType(Enum):
    #Degree_centrality
    DEGREES = 0
    AVERAGE_DEGREE = 1
    #In betweeness
    BETWEENNESS = 2
    AVERAGE_BETWEENNESS = 3
    #Excentricity
    ECCENTRICITY = 4
    AVERAGE_ECCENTRICITY = 5
    #Page rank
    PAGE_RANK = 6
    AVERAGE_PAGE_RANK = 7
    #Clustering_Coefficient
    CLUSTERING_COEFFICIENTS = 8
    AVERAGE_CLUSTERING_COEFFICIENTS = 9
    

class NetworkMetricCalculator:
    
    def __init__(self):
        self.maxNodeDepth = 15
        self.maxIter = 1000
    
    def computeDegrees(self, adjacencyMatrix, standardized):
        nNodes = np.size(adjacencyMatrix,0)
        degrees = np.zeros(nNodes)
        for i in tqdm(range(0,nNodes)):
            degrees[i] = sum(adjacencyMatrix[i,:])
        if(standardized):
            return degrees/(nNodes - 1)
        else:
            return degrees
    
    def computeDegreeDistribution(self, adjacencyMatrix):
        degrees = self.computeDegrees(adjacencyMatrix, False)
        maxValue = np.max(degrees)
        minValue = np.min(degrees)
        
        values = np.arange(minValue, maxValue+1, 1)
        
        degreeDistribution = np.zeros(int(maxValue - minValue + 1))
        for i in range(0,len(degrees)):
            degreeValue = degrees[i]
            distIndex = int(degreeValue - minValue)
            degreeDistribution[distIndex] = degreeDistribution[distIndex] + 1
        
        return values, degreeDistribution/sum(degreeDistribution)
    
    def getNumberOfShortestestPathsBetweenNodesFast(self, node1Index, node2Index, adjacencyMatrix):
        exploredNodes = []
        nodesToExplore = []
        nodesToExplore.append(node1Index)
        nNodes = np.size(adjacencyMatrix, 0)
        parents = np.ones(nNodes)*(-1)
        maxIters = 10
        nIters = 0
        added = True
        minDist = self.getMinDistBetweenNodesFast(node1Index, node2Index, adjacencyMatrix)
        nPaths = 0
        while(len(nodesToExplore) > 0 and nIters < maxIters and added):
            nextArr = []
            added = False
            for i in range(0,nNodes):
                if(parents[i] == -1):
                    neighbors = np.where(adjacencyMatrix[i,:] == 1)[0]
                    for j in neighbors:
                        if(j in nodesToExplore):
                            parents[i] = j
                            nextArr.append(i)
                            added = True
                            if(i == node2Index):
                                dst = 0
                                currentIndex = node2Index
                                while(not currentIndex == node1Index):
                                    currentIndex = parents[int(currentIndex)]
                                    dst = dst + 1
                                
                                if(dst == minDist):
                                    nPaths = nPaths + 1
                                    
                            
            exploredNodes.extend(nodesToExplore)
            nIters = nIters + 1
            nodesToExplore = nextArr
        
        return nPaths
    
    def getNumberOfShortestPathsPassingThroughNodeFast(self, node1Index, node2Index, nodeBetweenIndex, adjacencyMatrix):
        exploredNodes = []
        nodesToExplore = []
        nodesToExplore.append(node1Index)
        nNodes = np.size(adjacencyMatrix, 0)
        parents = np.ones(nNodes)*(-1)
        maxIters = 10
        nIters = 0
        added = True
        minDist = self.getMinDistBetweenNodesFast(node1Index, node2Index, adjacencyMatrix)
        nPaths = 0
        nPathsBetween = 0
        while(len(nodesToExplore) > 0 and nIters < maxIters and added):
            nextArr = []
            added = False
            for i in range(0,nNodes):
                if(parents[i] == -1):
                    neighbors = np.where(adjacencyMatrix[i,:] == 1)[0]
                    for j in neighbors:
                        if(j in nodesToExplore):
                            parents[i] = j
                            if(i != node2Index):
                                nextArr.append(i)
                                added = True
                            if(i == node2Index):
                                dst = 0
                                currentIndex = node2Index
                                nodeBetween = False
                                while(not currentIndex == node1Index):
                                    currentIndex = parents[int(currentIndex)]
                                    dst = dst + 1
                                    if(currentIndex == nodeBetweenIndex):
                                        nodeBetween = True
                                
                                if(dst == minDist):
                                    nPaths = nPaths + 1
                                    if(nodeBetween):
                                        nPathsBetween = nPathsBetween + 1
                                    
                            
            exploredNodes.extend(nodesToExplore)
            nIters = nIters + 1
            nodesToExplore = nextArr
        
        return nPaths, nPathsBetween
    
    #We'll do a fast bottom up version
    def getMinDistBetweenNodesFast(self, node1Index, node2Index, adjacencyMatrix):
        exploredNodes = []
        nodesToExplore = []
        nodesToExplore.append(node1Index)
        nNodes = np.size(adjacencyMatrix, 0)
        parents = np.ones(nNodes)*(-1)
        foundNode = False
        maxIters = 10
        nIters = 0
        added = True
        while(len(nodesToExplore) > 0 and not foundNode and nIters < maxIters and added):
            nextArr = []
            added = False
            for i in range(0,nNodes):
                if(parents[i] == -1):
                    neighbors = np.where(adjacencyMatrix[i,:] == 1)[0]
                    for j in neighbors:
                        if(j in nodesToExplore):
                            parents[i] = j
                            nextArr.append(i)
                            added = True
                            if(i == node2Index):
                                foundNode = True
                                break
                            
            exploredNodes.extend(nodesToExplore)
            nIters = nIters + 1
            nodesToExplore = nextArr
        
        if(foundNode):
            dst = 0
            currentIndex = node2Index
            while(not currentIndex == node1Index):
                currentIndex = parents[int(currentIndex)]
                dst = dst + 1
            return dst
        
        return 0

    
    
    def getBetweenessCentralities(self, adjacencyMatrix):
        nNodes = np.size(adjacencyMatrix,0)
        betweenessValues = np.zeros(nNodes)
        for i in tqdm(range(0,nNodes)):
            betweeness = 0
            for l in range(0,nNodes):
                for s in range(0,nNodes):
                    if(l != s and i!= l and i != s):
                        nShortestPaths, nShortestPathsBetween = self.getNumberOfShortestPathsPassingThroughNodeFast(l, s, i, adjacencyMatrix)
                        
                        #print("Shortest paths between "+str(l)+" and "+str(s) +" is "+str(nShortestPaths))
                        #print("Shortest paths where "+str(i) + " is in the middle "+str(nShortestPathsBetween))
                        #print("Shortest distance between "+str(l) +" and "+str(s) + " is "+str(self.getMinDistBetweenNodes(l,s, adjacencyMatrix)))
                        if(nShortestPaths != 0):
                            betweeness = betweeness + nShortestPathsBetween/nShortestPaths
            
            betweenessValues[i] = betweeness/((nNodes - 1)*(nNodes - 2))
        
        return betweenessValues
        
        
    def getEccentricities(self, adjacencyMatrix):
        nNodes = np.size(adjacencyMatrix, 0)
        eccentricities = np.zeros(nNodes)
        for i in tqdm(range(0,nNodes)):
            distances = []
            for j in range(0,nNodes):
                if(i != j):
                    dst = self.getMinDistBetweenNodesFast(i,j,adjacencyMatrix)
                    distances.append(dst)
            
            eccentricities[i] = np.max(distances)
        
        return eccentricities
        
    
    
    def getClusteringCoefficients(self, adjacencyMatrix):
        nNodes = np.size(adjacencyMatrix,0)
        clusteringCoefficients = np.zeros(nNodes)
        for i in tqdm(range(0,nNodes)):
            #We first get the neighborhood 
            neighbors = self.getNeighbors(i, adjacencyMatrix)
            connectionsSum = 0
            for s in neighbors:
                for l in neighbors:
                    if(s != l):
                        connectionsSum = connectionsSum + adjacencyMatrix[s,l]
            k = len(neighbors)
            if(k > 1):
                clusteringCoeff = connectionsSum/(k*(k-1))
                clusteringCoefficients[i] = clusteringCoeff
        
        return clusteringCoefficients
                
    
    
    def getNeighbors(self, index, adjacencyMatrix):
        
        nNodes = np.size(adjacencyMatrix,0)
        neighbors = []
        for i in range(0,nNodes):
            if(i != index and (adjacencyMatrix[index,i] == 1 or adjacencyMatrix[i,index] == 1)):
                neighbors.append(i)
        
        return neighbors
    
    #k is the number of steps
    def getPageRanks(self, k, adjacencyMatrix):
        nNodes = np.size(adjacencyMatrix, 0)
        pageRanks = np.ones(nNodes)
        pageRanks = pageRanks*(1/nNodes)
        
        for i in tqdm(range(0,k)):
            oldRanks = pageRanks.copy()
            for j in range(0,nNodes):
                rank = 0
                for s in range(0,nNodes):
                    if(s != j and adjacencyMatrix[s, j] == 1):
                        nPieces = sum(adjacencyMatrix[s,:])
                        rank = rank + (1/nPieces)*oldRanks[s]
                
                pageRanks[j] = rank
            
        return pageRanks
    
    
    def computeNetworkMetric(self, adjacencyMatrix, metricType):
        if(metricType == NetworkMetricType.DEGREES):
            degrees =  self.computeDegrees(adjacencyMatrix, True)
            clear_output(wait = True)
            return degrees
        
        elif(metricType == NetworkMetricType.AVERAGE_DEGREE):
            meanDegree = np.mean(self.computeDegrees(adjacencyMatrix, True))
            clear_output(wait = True)
            return meanDegree
        
        elif(metricType == NetworkMetricType.BETWEENNESS):
            betweennessValues = self.getBetweenessCentralities(adjacencyMatrix)
            clear_output(wait = True)
            return betweennessValues
        
        elif(metricType == NetworkMetricType.AVERAGE_BETWEENNESS):
            averageBetweenness =  np.mean(self.getBetweenessCentralities(adjacencyMatrix))
            clear_output(wait = True)
            return averageBetweenness
        
        elif(metricType == NetworkMetricType.ECCENTRICITY):
            eccentricities = self.getEccentricities(adjacencyMatrix)
            clear_output(wait = True)
            return eccentricities
        
        elif(metricType == NetworkMetricType.AVERAGE_ECCENTRICITY):
            averageEccentricity = np.mean(self.getEccentricities(adjacencyMatrix))
            clear_output(wait = True)
            return averageEccentricity
        
        elif(metricType == NetworkMetricType.PAGE_RANK):
            pageRanks = self.getPageRanks(20,adjacencyMatrix)
            clear_output(wait = True)
            return pageRanks
        
        elif(metricType == NetworkMetricType.AVERAGE_PAGE_RANK):
            averagePageRanks = np.mean(self.getPageRanks(20,adjacencyMatrix))
            clear_output(wait = True)
            return averagePageRanks
        
        elif(metricType == NetworkMetricType.CLUSTERING_COEFFICIENTS):
            clusteringCoeffs = self.getClusteringCoefficients(adjacencyMatrix)
            clear_output(wait = True)
            return clusteringCoeffs
        
        elif(metricType == NetworkMetricType.AVERAGE_CLUSTERING_COEFFICIENTS):
            averageClusteringCoeffs = np.mean(self.getClusteringCoefficients(adjacencyMatrix))
            clear_output(wait = True)
            return averageClusteringCoeffs
        
    
    def loadAdjacencyMatrix(self, filePath):
        adjacencyMatrix = []
        with open(filePath) as csvfile:
            csvReader = csv.reader(csvfile, delimiter=',')     
            for row in csvReader:
                rowArray = []
                for i in range(0,len(row)):
                    value = int(row[i])
                    rowArray.append(value)
                
                adjacencyMatrix.append(rowArray)
        
        return np.array(adjacencyMatrix)
    
    
    def drawGraph(self, adjacencyMatrix, nIters, scale):
        nNodes = np.size(adjacencyMatrix,0)
        nodePositions = np.zeros((nNodes, 2))
        c1 = 0.5
        c2 = 5
        c3 = 0.01
        c4 = 0.005
        for i in range(0,nNodes):
            nodePositions[i,:] = [random.random()*scale, random.random()*scale]
        
        for i in tqdm(range(0,nIters)):
            positionsTemp = nodePositions.copy()
            for l in range(0,nNodes):
                nodePosition1 = nodePositions[l,:]
                force = np.zeros(2)
                for s in range(0,nNodes):
                    nodePosition2 = nodePositions[s,:]
                    #Fuerza de atracción
                    if(adjacencyMatrix[l,s] == 1):
                        dst = np.sqrt((nodePosition1[0] - nodePosition2[0])**2 + (nodePosition1[1] - nodePosition2[1])**2)
                        if(dst != 0):
                            forceMag = c1*np.log(dst/c2)
                            forceComponent = (np.array([nodePosition2[0] - nodePosition1[0], nodePosition2[1] - nodePosition1[1]])/dst)*forceMag
                            force = force + forceComponent
                    
                    else:
                        
                        dst = np.sqrt((nodePosition1[0] - nodePosition2[0])**2 + (nodePosition1[1] - nodePosition2[1])**2)
                        if(dst != 0):
                            forceMag = (c3/(dst**2))
                            forceComponent = (np.array([nodePosition1[0] - nodePosition2[0], nodePosition1[1] - nodePosition2[1]])/dst)*forceMag
                            force = force + forceComponent
                    
                positionsTemp[l,:] = positionsTemp[l,:] + (1/(i+1))*c4*force
            
            nodePositions = positionsTemp
        
        #Draw the connections
        for i in range(0,nNodes):
            for j in range(0,nNodes):
                if(adjacencyMatrix[i,j] == 1):
                    plt.plot([nodePositions[i,0], nodePositions[j,0]], [nodePositions[i,1], nodePositions[j,1]], color = "#ff870f", linewidth = 2)
        
        
        #Draw the nodes
        for i in range(0,nNodes):
            plt.plot(nodePositions[i,0], nodePositions[i,1], marker = "o", linestyle = "none", color = "#16bdf5", markersize = 20)
            plt.text(nodePositions[i,0], nodePositions[i,1], str(i), horizontalalignment='center',verticalalignment='center')
    
    
    
    
    def drawGraphFruchterman(self, adjacencyMatrix, nIters, width, height, labelNodes):
        
        nNodes = np.size(adjacencyMatrix,0)
        nodePositions = np.zeros((nNodes, 2))
        nodeDisp = np.zeros((nNodes, 2))
        area = width*height
        C = 1
        k = C*np.sqrt(area/nNodes)
        t = 10
        
        for i in range(0,nNodes):
            nodePositions[i,:] = [random.random()*height, random.random()*width]
        
        for s in tqdm(range(0,nIters)):
        
            #Repulsive forces
            for i in range(0,nNodes):
                nodeDisp[i,:] = np.array([0,0])
                for j in range(0,nNodes):
                    if(i != j):
                        delta = np.array([nodePositions[i,0] - nodePositions[j,0], nodePositions[i,1] - nodePositions[j,1]])
                        deltaNorm = np.sqrt((delta[1])**2 + (delta[0])**2)
                        if(deltaNorm != 0):
                            repulsiveForce = (k**2)/deltaNorm
                            nodeDisp[i,:] = nodeDisp[i,:] + repulsiveForce*(delta/deltaNorm)
        
            #Attractive forces with the edges
            for i in range(0,nNodes):
                for j in range(0,nNodes):
                    if(i != j and adjacencyMatrix[i,j] == 1):
                        delta = np.array([nodePositions[i,0] - nodePositions[j,0], nodePositions[i,1] - nodePositions[j,1]])
                        deltaNorm = np.sqrt((delta[1])**2 + (delta[0])**2)
                        if(deltaNorm != 0):
                            attractiveForce = -(deltaNorm**2)/k
                            nodeDisp[i,:] = nodeDisp[i,:] + attractiveForce*(delta/deltaNorm)
            
            #Adjust the positions
            for i in range(0,nNodes):
                normDisp = np.sqrt(nodeDisp[i,0]**2 + nodeDisp[i,1]**2)
                nodePositions[i,:] = nodePositions[i,:] + (nodeDisp[i,:]/normDisp)*np.min([normDisp, t])
                nodePositions[i,1] = np.min([width/2, np.max([nodePositions[i,1], -width/2])])
                nodePositions[i,0] = np.min([height/2, np.max([nodePositions[i,0], -height/2])])
            
            t = (1/(s + 1))*t 
                
                        
        #Draw the connections
        for i in range(0,nNodes):
            for j in range(0,nNodes):
                if(adjacencyMatrix[i,j] == 1):
                    plt.plot([nodePositions[i,0], nodePositions[j,0]], [nodePositions[i,1], nodePositions[j,1]], color = "#ff870f", linewidth = 2)
        
        
        #Draw the nodes
        for i in range(0,nNodes):
            plt.plot(nodePositions[i,0], nodePositions[i,1], marker = "o", linestyle = "none", color = "#16bdf5", markersize = 20)
            if(labelNodes):
                plt.text(nodePositions[i,0], nodePositions[i,1], str(i), horizontalalignment='center',verticalalignment='center')
        
                
                        
                
            
        
        
        
            
        
        
        

        
    
    
            
        
        