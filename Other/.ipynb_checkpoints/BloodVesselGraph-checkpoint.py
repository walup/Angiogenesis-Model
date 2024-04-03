from skimage.morphology import skeletonize, thin,medial_axis
import random 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Connection:
    
    def __init__(self, fromNodeId, toNodeId, weight):
        self.fromNodeId = fromNodeId
        self.toNodeId = toNodeId
        self.weight = weight
    
    def __eq__(self, other):
        return not (other == None) and self.fromNodeId == other.fromNodeId and self.toNodeId == other.toNodeId
        
class Node:
    def __init__(self, nodeId, automatonIndex1, automatonIndex2):
        self.nodeId = nodeId
        self.automatonIndex1 = automatonIndex1
        self.automatonIndex2 = automatonIndex2
    
    def __eq__(self,other):
        self.nodeId == other.nodeId or( self.automatonIndex1 == other.automatonIndex1 and self.automatonIndex2 == other.automatonIndex2)
        

class GraphSketchType:
    ORIGINAL = 0
    CIRCULAR = 1
    AESTHETIC = 2
    
    
        
class BloodVesselGraph:
    
    def __init__(self):
        self.nodes = []
        self.connections = {}
        self.nodesColor = [39/255, 162/255, 219/255]
        self.connectionsColor = [201/255, 74/255, 0/255]
        self.specialTumorNode = False
        self.tumorNodeColor = [58/255.0, 252/255.0, 84/255.0]
        self.differenceRadius = 7
        
    
    def isNotCloseToOtherNodes(self, node):
        for i in range(0,len(self.nodes)):
            dst = np.abs(node.automatonIndex1 - self.nodes[i].automatonIndex1) + np.abs(node.automatonIndex2 - self.nodes[i].automatonIndex2)
            if(dst < self.differenceRadius):
                return False
        
        return True
    
    
    def addNode(self, node):
        if(not node in self.nodes and self.isNotCloseToOtherNodes(node)):
            self.nodes.append(node)
            self.connections[node.nodeId] = []
            return True
        
        return False
            
    
    def nodeExists(self, nodeId):
        for i in range(0,len(self.nodes)):
            if(self.nodes[i].nodeId == nodeId):
                return True
        
        return False
    
    def hasConnection(self, startNodeId, endNodeId):
        connections = self.connections[startNodeId]
        for i in range(0,len(connections)):
            if(connections[i].toNodeId == endNodeId):
                return True
        
        return False
    
    
    def addConnection(self, fromNodeId, toNodeId, weight):
        if(self.nodeExists(fromNodeId)):
            newConnection = Connection(fromNodeId, toNodeId, weight)
            if(not newConnection in self.connections[fromNodeId]):
                self.connections[fromNodeId].append(newConnection)
    
    def getAllConnections(self):
        allConnections = []
        for i in range(0,len(self.nodes)):
            node = self.nodes[i]
            connections = self.connections[node.nodeId]
            for j in range(0,len(connections)):
                connection = connections[j]
                allConnections.append(connection)
        
        return allConnections
    
    def getNodeById(self, nodeId):
        for i in range(0,len(self.nodes)):
            if(self.nodes[i].nodeId == nodeId):
                return self.nodes[i]
        
        return None
    
    def getNodeIndexById(self, nodeId):
        for i in range(0,len(self.nodes)):
            if(self.nodes[i].nodeId == nodeId):
                return i
        
        return None
    
    def getNodeByLocation(self, index1, index2):
        for i in range(0,len(self.nodes)):
            if(self.nodes[i].automatonIndex1 == index1 and self.nodes[i].automatonIndex2 == index2):
                return self.nodes[i]
        
        return None
    
    def drawGraph(self, scale, ax, graphType):
        #Draw the connections
        if(not self.specialTumorNode):
            if(graphType == GraphSketchType.ORIGINAL):
                allConnections = self.getAllConnections()
        
                for i in range(0,len(allConnections)):
                    connection = allConnections[i]
                    fromNode = self.getNodeById(connection.fromNodeId)
                    toNode = self.getNodeById(connection.toNodeId)
                    ax.plot([fromNode.automatonIndex2*scale, toNode.automatonIndex2*scale], [fromNode.automatonIndex1*scale, toNode.automatonIndex1*scale], color = self.connectionsColor, linewidth = 1, alpha = 0.5)
            
        
                for i in range(0,len(self.nodes)):
                    node = self.nodes[i]
                    ax.plot(node.automatonIndex2*scale, node.automatonIndex1*scale, color = self.nodesColor, linestyle = "none", marker = "o", markersize = 1, alpha = 0.5)
                
            elif(graphType == GraphSketchType.CIRCULAR):
                circularXPositions = np.zeros(len(self.nodes))
                circularYPositions = np.zeros(len(self.nodes))
                shuffleIndexes= list(range(0, len(self.nodes)))
                random.shuffle(shuffleIndexes)
                angleDelta = 2*np.pi/len(self.nodes)
                radius = 10*scale
            
                #Draw the nodes
                for i in range(0,len(self.nodes)):
                    xCirclePosition = radius*np.cos(angleDelta*i)
                    yCirclePosition = radius*np.sin(angleDelta*i)
                    circularXPositions[shuffleIndexes[i]] = xCirclePosition
                    circularYPositions[shuffleIndexes[i]] = yCirclePosition
            
                ax.plot(circularXPositions, circularYPositions, color = self.nodesColor, marker = "o", linestyle = "none", markersize = 0.5)
            
                #Draw the connections
                connections = self.getAllConnections()
                print(len(connections))
                for i in range(0,len(connections)):
                    connection = connections[i]
                    fromNodeId = connection.fromNodeId
                    toNodeId = connection.toNodeId
                    #We need the indexes of the nodes to draw the connectino
                    fromNodeIndex = self.getNodeIndexById(fromNodeId)
                    toNodeIndex = self.getNodeIndexById(toNodeId)
                    #Draw the connections
                
                    ax.plot([circularXPositions[fromNodeIndex], circularXPositions[toNodeIndex]], [circularYPositions[fromNodeId],circularYPositions[toNodeId]], linewidth = 0.1, alpha = 0.5, color = self.connectionsColor)
            elif(graphType == GraphSketchType.AESTHETIC):
                positions = np.zeros((len(self.nodes),2))
                c1 = 0.5
                c2 = 5
                c3 = 0.01
                c4 = 0.005
                iterations = 30

                #initialize the positions as random
                for i in range(0,len(self.nodes)):
                    positions[i,:] = [random.random()*scale, random.random()*scale]
            
                #Now iterate 10 times calcuolating forces and moving the nodes
                for i in tqdm(range(0,iterations)):
                    positionsTemp = positions.copy()
                    for l in range(0,len(self.nodes)):
                        node1 = self.nodes[l]
                        position1 = positions[l,:]
                        force = np.zeros(2)
                        for m in range(0,len(self.nodes)):
                            #We see if the node is connected to the one we are examining
                            node2 = self.nodes[m]
                            position2 = positions[m,:]
                            if(node2.nodeId in [x.toNodeId for x in self.connections[node1.nodeId]]):
                                dst = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
                                if(dst != 0):
                                    forceMagnitude = c1*np.log(dst/c2)
                                    forceComponent = (np.array([position2[0] - position1[0], position2[1] - position1[1]])/dst)*forceMagnitude
                                    force = force + forceComponent
                            
                            else:
                                dst = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
                                if(dst!= 0):
                                    forceMagnitude = (c3/(dst**2))
                                    forceComponent = ((np.array([position1[0] - position2[0], position1[1] - position2[1]]))/dst)*forceMagnitude
                                    force = force + forceComponent
                    
                        positionsTemp[l,:] = positionsTemp[l,:] + (1/(i+1))*c4*force
                    positions = positionsTemp
            
                #We draw the nodes
                ax.plot(positions[:,0], positions[:,1], color = self.nodesColor, marker = "o", linestyle = "none", markersize = 1)
            
                #We draw the connections
                for i in range(0,len(self.nodes)):
                    node = self.nodes[i]
                    connections = self.connections[node.nodeId]
                    nodeIndex1 = self.getNodeIndexById(node.nodeId)
                    position1 = positions[nodeIndex1,:]
                    for j in range(0,len(connections)):
                        nodeIndex2 = self.getNodeIndexById(connections[j].toNodeId)
                        position2 = positions[nodeIndex2,:]
                        ax.plot([position1[0],position2[0]], [position1[1], position2[1]],color = self.connectionsColor, linewidth = 0.2)
        
        
        else:
            
            if(graphType == GraphSketchType.ORIGINAL):
                allConnections = self.getAllConnections()
        
                for i in range(0,len(allConnections)):
                    connection = allConnections[i]
                    fromNode = self.getNodeById(connection.fromNodeId)
                    toNode = self.getNodeById(connection.toNodeId)
                    ax.plot([fromNode.automatonIndex2*scale, toNode.automatonIndex2*scale], [fromNode.automatonIndex1*scale, toNode.automatonIndex1*scale], color = self.connectionsColor, linewidth = 1, alpha = 0.5)
            
        
                for i in range(0,len(self.nodes)-1):
                    node = self.nodes[i]
                    ax.plot(node.automatonIndex2*scale, node.automatonIndex1*scale, color = self.nodesColor, linestyle = "none", marker = "o", markersize = 1, alpha = 0.5)
                
                ax.plot(self.nodes[len(self.nodes)-1].automatonIndex2*scale, self.nodes[len(self.nodes) - 1].automatonIndex1*scale, color = self.tumorNodeColor, marker = "o", markersize = 10, alpha = 0.5)
                
                
                ax.invert_yaxis()
                
            elif(graphType == GraphSketchType.CIRCULAR):
                circularXPositions = np.zeros(len(self.nodes))
                circularYPositions = np.zeros(len(self.nodes))
                shuffleIndexes= list(range(0, len(self.nodes)))
                random.shuffle(shuffleIndexes)
                angleDelta = 2*np.pi/len(self.nodes)
                radius = 10*scale
                
                for i in range(0,len(self.nodes)):
                    xCirclePosition = radius*np.cos(angleDelta*i)
                    yCirclePosition = radius*np.sin(angleDelta*i)
                    circularXPositions[shuffleIndexes[i]] = xCirclePosition
                    circularYPositions[shuffleIndexes[i]] = yCirclePosition
        
                connections = self.getAllConnections()
                print(len(connections))
                for i in range(0,len(connections)):
                    connection = connections[i]
                    fromNodeId = connection.fromNodeId
                    toNodeId = connection.toNodeId
                    #We need the indexes of the nodes to draw the connectino
                    fromNodeIndex = self.getNodeIndexById(fromNodeId)
                    toNodeIndex = self.getNodeIndexById(toNodeId)
                    #Draw the connections
                    ax.plot([circularXPositions[fromNodeIndex], circularXPositions[toNodeIndex]], [circularYPositions[fromNodeId],circularYPositions[toNodeId]], linewidth = 0.1, alpha = 0.5, color = self.connectionsColor)
            
            
                ax.plot(circularXPositions[0:len(self.nodes) - 1], circularYPositions[0:len(self.nodes) - 1], color = self.nodesColor, marker = "o", linestyle = "none", markersize = 0.5)
                ax.plot(circularXPositions[len(self.nodes) - 1], circularYPositions[len(self.nodes) - 1], color = self.tumorNodeColor, marker = "o", markersize = 10, alpha = 0.5)
            
            elif(graphType == GraphSketchType.AESTHETIC):
                nNodes = np.size(self.nodes)
                width = nNodes
                height = nNodes
                C = 1
                k = C*np.sqrt(((width)*(height))/nNodes)
                nodeDisp = np.zeros((nNodes,2))
                nodePositions = np.zeros((nNodes,2))
                t = nNodes
                nIters = 100
                
                for i in range(0,nNodes):
                    nodePositions[i,:] = np.array([random.random()*height, random.random()*width])
                
                for s in tqdm(range(0,nIters)):
                
                    #Attractive force
                    for i in range(0,nNodes):
                        nodeDisp[i,:] = np.array([0,0])
                        node1Id = self.nodes[i].nodeId
                        nodeConnectionIds = [x.toNodeId for x in self.connections[node1Id]]
                        for j in range(0,nNodes):
                            node2Id = self.nodes[j].nodeId
                            if(i != j and node2Id in nodeConnectionIds):
                                delta = np.array([nodePositions[i,0] - nodePositions[j,0], nodePositions[i,1] - nodePositions[j,1]])
                                deltaNorm = np.sqrt(delta[0]**2 + delta[1]**2)
                                if(deltaNorm != 0):
                                    attractiveForce = -(deltaNorm**2)/k
                                    nodeDisp[i,:] = nodeDisp[i,:] + attractiveForce*(delta/deltaNorm)
                
                    #Repulsive force
                    for i in range(0,nNodes):
                        for j in range(0,nNodes):
                            if(i != j):
                                delta = np.array([nodePositions[i,0] - nodePositions[j,0], nodePositions[i,1] - nodePositions[j,1]])
                                deltaNorm = np.sqrt((delta[1])**2 + (delta[0])**2)
                                if(deltaNorm != 0):
                                    repulsiveForce = (k**2)/deltaNorm
                                    nodeDisp[i,:] = nodeDisp[i,:] + repulsiveForce*(delta/deltaNorm)
                
                
                    for i in range(0,nNodes):
                        normDisp = np.sqrt(nodeDisp[i,0]**2 + nodeDisp[i,1]**2)
                        nodePositions[i,:] = nodePositions[i,:] + (nodeDisp[i,:]/normDisp)*np.min([normDisp, t])
                        #nodePositions[i,1] = np.min([width/2, np.max([nodePositions[i,1], -width/2])])
                        #nodePositions[i,0] = np.min([height/2, np.max([nodePositions[i,0], -height/2])])
                    
                    t = (1/(s + 1))*t 
                    
                    
                
            
                #We draw the connections
                for i in range(0,len(self.nodes)):
                    node = self.nodes[i]
                    connections = self.connections[node.nodeId]
                    nodeIndex1 = self.getNodeIndexById(node.nodeId)
                    position1 = nodePositions[nodeIndex1,:]
                    for j in range(0,len(connections)):
                        nodeIndex2 = self.getNodeIndexById(connections[j].toNodeId)
                        position2 = nodePositions[nodeIndex2,:]
                        ax.plot([position1[0],position2[0]], [position1[1], position2[1]],color = self.connectionsColor, linewidth = 0.1)
            
                #We draw the nodes
                ax.plot(nodePositions[0:len(self.nodes) - 1,0], nodePositions[0:len(self.nodes)-1,1], color = self.nodesColor, marker = "o", linestyle = "none", markersize = 2, alpha = 0.5)
                ax.plot(nodePositions[len(self.nodes)-1,0], nodePositions[len(self.nodes) - 1, 1], color = self.tumorNodeColor, marker = "o", markersize = 10, alpha = 0.5)
            
            
                        
            
            
        
        
    
    def getDegreeDistribution(self):
        maxDegree = len(self.nodes)
        degreeDistribution = np.zeros(maxDegree + 1)
        xValues = list(range(0,len(self.nodes) + 1))
        for i in range(0,len(self.nodes)):
            nodeConnections = self.connections[self.nodes[i].nodeId]
            nodeDegree = len(nodeConnections)
            degreeDistribution[nodeDegree] = degreeDistribution[nodeDegree]+1
        
        return xValues, degreeDistribution/sum(degreeDistribution)
    
    
    def getAverageDegree(self):
        xValues, degreeDistribution = self.getDegreeDistribution()
        avg = 0
        for i in range(0,len(xValues)):
            avg = avg + xValues[i]*degreeDistribution[i]
        
        return avg
    
    def exportAdjacencyTable(self, fileName):
        s = ""
        for i in range(0,len(self.nodes)):
            nodeConnections = self.connections[self.nodes[i].nodeId]
            nodeConnectionIds = [x.toNodeId for x in nodeConnections]
            for j in range(0,len(self.nodes)):
                if(self.nodes[j].nodeId in nodeConnectionIds):
                    s = s + str(1)       
                else:
                    s = s + str(0)
                
                if(j != len(self.nodes) - 1):
                    s = s + ","
                   
                elif(i != len(self.nodes) - 1):
                    s = s + "\n"
        file = open(fileName, "w")
        file.write(s)
        file.close()
                
        
                
              
                
class AutomatonToGraph:
    
    def __init__(self):
        self.nodeNeighborsThreshold = 4
        self.differenceRadius = 0
        
        
    def convertAutomatonToGraphWithTumor(self, vesselLocations, proliferatingLocations):
        
        vesselLocations = skeletonize(vesselLocations)
        self.skeletonizedImage = vesselLocations
        graph = BloodVesselGraph()
        graph.differenceRadius = self.differenceRadius
        nodeEdgeMatrix = np.zeros((np.size(vesselLocations,0), np.size(vesselLocations,1)))
        nodeCount = 0
        
        for i in range(1, np.size(vesselLocations,0)-1):
            for j in range(1,np.size(vesselLocations,1)-1):
                nNeighbors = int(vesselLocations[i-1,j]) + int(vesselLocations[i+1,j]) + int(vesselLocations[i,j+1]) + int(vesselLocations[i,j-1])
                
                #Candidates of being branching points 
                if(vesselLocations[i,j] == 1 and nNeighbors == 3 or nNeighbors == 4):
                    newNode = Node(nodeCount, i, j)
                    nodeAdded = graph.addNode(newNode)
                    if(nodeAdded):
                        nodeEdgeMatrix[i,j] = 2
                        nodeCount = nodeCount + 1
                    else:
                        nodeEdgeMatrix[i,j] = 1
                elif(vesselLocations[i,j] == 1 and nNeighbors == 2):
                    newNode = Node(nodeCount, i, j)
                    nodeAdded = graph.addNode(newNode)
                    if(nodeAdded):
                        nodeEdgeMatrix[i,j] = 2
                        nodeCount = nodeCount + 1
                    else:
                        nodeEdgeMatrix[i,j] = 1
                        
                elif(vesselLocations[i,j] == 1):
                    nodeEdgeMatrix[i,j] = 1
        
        tumorIndex1 = int(np.size(vesselLocations,0)/2)
        tumorIndex2 = int(np.size(vesselLocations,1)/2)
        tumorNode = Node(nodeCount, tumorIndex1, tumorIndex2)
        graph.addNode(tumorNode)
        
        
        
        capIters = 200
        #BFS
        for i in tqdm(range(0,len(graph.nodes)-1)):
            nodesToExplore = []
            nodesToExplore.append([graph.nodes[i].automatonIndex1, graph.nodes[i].automatonIndex2])
            iterNum = 0
            exploredNodes = []
            added = True
            while(len(nodesToExplore) > 0 and iterNum < capIters):
                nodeToExplore = nodesToExplore.pop(0)
                for l in range(-1,2):
                    for s in range(-1,2):
                        if(not (l == 0 and s == 0) and nodeToExplore[0] + l >= 0 and nodeToExplore[0] + l < np.size(nodeEdgeMatrix,0) and nodeToExplore[1] + s >= 0 and nodeToExplore[1] + s < np.size(nodeEdgeMatrix,1) and nodeEdgeMatrix[nodeToExplore[0] + l, nodeToExplore[1] + s] == 1 and not [nodeToExplore[0] + l, nodeToExplore[1]+s] in exploredNodes):
                            nodesToExplore.append([nodeToExplore[0] + l, nodeToExplore[1] + s])
                        
                        elif(not (l == 0 and s == 0) and nodeToExplore[0] + l >= 0 and nodeToExplore[0] + l < np.size(nodeEdgeMatrix,0) and nodeToExplore[1] + s >= 0 and nodeToExplore[1] + s < np.size(nodeEdgeMatrix,1) and nodeEdgeMatrix[nodeToExplore[0] + l, nodeToExplore[1] + s] == 2):
                            if(graph.getNodeByLocation(nodeToExplore[0] + l, nodeToExplore[1] + s).nodeId != graph.nodes[i].nodeId):
                                graph.addConnection(graph.nodes[i].nodeId, graph.getNodeByLocation(nodeToExplore[0] + l, nodeToExplore[1] + s).nodeId, 1)
                
                exploredNodes.append(nodeToExplore)
                iterNum = iterNum + 1
            
            prolifNeighbors = 0
            for l in range(-1,2):
                for s in range(-1,2):
                    index1 =graph.nodes[i].automatonIndex1 + l
                    index2 = graph.nodes[i].automatonIndex2 + s
                    if(index1 >= 0 and index1 < np.size(proliferatingLocations,0) and index2 >= 0 and index2 < np.size(proliferatingLocations,1)):
                            if(proliferatingLocations[index1, index2] == 1):
                                prolifNeighbors = prolifNeighbors + 1
            
            if(prolifNeighbors > 0):
                graph.addConnection(graph.nodes[i].nodeId, nodeCount, 1)
                graph.addConnection(nodeCount, graph.nodes[i].nodeId,1)
        
        graph.specialTumorNode = True
                
                                
        
        return graph
        
        
    
    def convertAutomatonToGraph(self, vesselLocations):
        vesselLocations = skeletonize(vesselLocations)
        self.skeletonizedImage = vesselLocations
        
        graph = BloodVesselGraph()
        graph.differenceRadius = self.differenceRadius
        nodeEdgeMatrix = np.zeros((np.size(vesselLocations,0), np.size(vesselLocations,1)))
        nodeCount = 0
        
        for i in range(1, np.size(vesselLocations,0)-1):
            for j in range(1,np.size(vesselLocations,1)-1):
                nNeighbors = int(vesselLocations[i-1,j]) + int(vesselLocations[i+1,j]) + int(vesselLocations[i,j+1]) + int(vesselLocations[i,j-1])
                
                #Candidates of being branching points 
                if(vesselLocations[i,j] == 1 and nNeighbors == 3 or nNeighbors == 4):
                    newNode = Node(nodeCount, i, j)
                    nodeAdded = graph.addNode(newNode)
                    if(nodeAdded):
                        nodeEdgeMatrix[i,j] = 2
                        nodeCount = nodeCount + 1
                    else:
                        nodeEdgeMatrix[i,j] = 1
                elif(vesselLocations[i,j] == 1 and nNeighbors == 2):
                    newNode = Node(nodeCount, i, j)
                    nodeAdded = graph.addNode(newNode)
                    if(nodeAdded):
                        nodeEdgeMatrix[i,j] = 2
                        nodeCount = nodeCount + 1
                    else:
                        nodeEdgeMatrix[i,j] = 1
                        
                elif(vesselLocations[i,j] == 1):
                    nodeEdgeMatrix[i,j] = 1
        
        capIters = 2000
        #BFS
        for i in tqdm(range(0,len(graph.nodes))):
            nodesToExplore = []
            nodesToExplore.append([graph.nodes[i].automatonIndex1, graph.nodes[i].automatonIndex2])
            iterNum = 0
            exploredNodes = []
            while(len(nodesToExplore) > 0 and iterNum < capIters):
                nodeToExplore = nodesToExplore.pop(0)
                for l in range(-1,2):
                    for s in range(-1,2):
                        if(not (l == 0 and s == 0) and nodeToExplore[0] + l >= 0 and nodeToExplore[0] + l < np.size(nodeEdgeMatrix,0) and nodeToExplore[1] + s >= 0 and nodeToExplore[1] + s < np.size(nodeEdgeMatrix,1) and nodeEdgeMatrix[nodeToExplore[0] + l, nodeToExplore[1] + s] == 1 and not [nodeToExplore[0] + l, nodeToExplore[1]+s] in exploredNodes):
                            nodesToExplore.append([nodeToExplore[0] + l, nodeToExplore[1] + s])
                        
                        elif(not (l == 0 and s == 0) and nodeToExplore[0] + l >= 0 and nodeToExplore[0] + l < np.size(nodeEdgeMatrix,0) and nodeToExplore[1] + s >= 0 and nodeToExplore[1] + s < np.size(nodeEdgeMatrix,1) and nodeEdgeMatrix[nodeToExplore[0] + l, nodeToExplore[1] + s] == 2):
                            if(graph.getNodeByLocation(nodeToExplore[0] + l, nodeToExplore[1] + s).nodeId != graph.nodes[i].nodeId):
                                graph.addConnection(graph.nodes[i].nodeId, graph.getNodeByLocation(nodeToExplore[0] + l, nodeToExplore[1] + s).nodeId, 1)
                        
                exploredNodes.append(nodeToExplore)
                
                                
        
        return graph
    
    
    
    def convertAutomatonToGraph8Dirs(self, vesselLocations):
        vesselLocations = skeletonize(vesselLocations)
        
        graph = BloodVesselGraph()
        nodeEdgeMatrix = np.zeros((np.size(vesselLocations,0), np.size(vesselLocations,1)))
        nodeCount = 0
    
        
        for i in range(1, np.size(vesselLocations,0)-1):
            for j in range(1,np.size(vesselLocations,1)-1):
                nNeighbors = int(vesselLocations[i-1,j]) + int(vesselLocations[i+1,j]) + int(vesselLocations[i,j+1]) + int(vesselLocations[i,j-1])
                
                if(vesselLocations[i,j] == 1 and nNeighbors == 3 or nNeighbors == 4):
                    newNode = Node(nodeCount, i, j)
                    graph.addNode(newNode)
                    nodeEdgeMatrix[i,j] = 2
                    nodeCount = nodeCount + 1
                elif(vesselLocations[i,j] == 1):
                    nodeEdgeMatrix[i,j] = 1
                    
                if(vesselLocations[i,j] == 1 and nNeighbors == 2):
                    newNode = Node(nodeCount, i, j)
                    graph.addNode(newNode)
                    nodeEdgeMatrix[i,j] = 2
                    nodeCount = nodeCount + 1
        
        capIters = 2000
        #We won't use BFS but a sort of branch explorer
        
        for i in tqdm(range(0,len(graph.nodes))):
            node = graph.nodes[i]
            index1 = node.automatonIndex1
            index2 = node.automatonIndex2
            parentLocation = [index1, index2]
            
            southEast = self.getShortestConnection(nodeEdgeMatrix, [index1 -1, index2 + 1], parentLocation, graph, node.nodeId)
            if(southEast != None):
                graph.addConnection(southEast.fromNodeId, southEast.toNodeId, southEast.weight)
            
            east = self.getShortestConnection(nodeEdgeMatrix, [index1, index2 + 1], parentLocation, graph, node.nodeId)
            if(east != None):
                graph.addConnection(east.fromNodeId, east.toNodeId, east.weight)
            
            northEast = self.getShortestConnection(nodeEdgeMatrix, [index1 -1, index2 - 1], parentLocation,graph, node.nodeId)
            if(northEast != None):
                graph.addConnection(northEast.fromNodeId, northEast.toNodeId, northEast.weight)
            
            southWest = self.getShortestConnection(nodeEdgeMatrix, [index1 -1, index2 - 1], parentLocation,graph, node.nodeId)
            if(southWest != None):
                graph.addConnection(southWest.fromNodeId, southWest.toNodeId, southWest.weight)
            
            west = self.getShortestConnection(nodeEdgeMatrix, [index1, index2 - 1], parentLocation,graph, node.nodeId)
            if(west != None):
                graph.addConnection(west.fromNodeId, west.toNodeId, west.weight)
            
            northWest= self.getShortestConnection(nodeEdgeMatrix, [index1 -1, index2 + 1], parentLocation,graph, node.nodeId)
            if(northWest != None):
                graph.addConnection(northWest.fromNodeId, northWest.toNodeId, northWest.weight)
            
            north = self.getShortestConnection(nodeEdgeMatrix, [index1 -1, index2], parentLocation,graph, node.nodeId)
            if(north != None):
                graph.addConnection(north.fromNodeId, north.toNodeId, north.weight)
            
            south = self.getShortestConnection(nodeEdgeMatrix, [index1 +1, index2], parentLocation,graph, node.nodeId)
            if(south != None):
                graph.addConnection(south.fromNodeId, south.toNodeId, south.weight)
        
        return graph
            
            
            
    
    def getShortestConnection(self, nodeEdgeMatrix, startLocation, parentLocation, graph, parentName):
        nodesToExplore = []
        nodesToExplore.append(startLocation)
        iterNum = 0
        exploredNodes = []
        capIters = 2000
        
        while(len(nodesToExplore) > 0 and iterNum < capIters):
            nodeToExplore = nodesToExplore.pop(0)
            for l in range(-1,2):
                for s in range(-1,2):
                    if(not (l == 0 and s == 0) and nodeToExplore[0] + l >= 0 and nodeToExplore[0] + l < np.size(nodeEdgeMatrix,0) and nodeToExplore[1] + s >= 0 and nodeToExplore[1] + s < np.size(nodeEdgeMatrix,1) and nodeEdgeMatrix[nodeToExplore[0] + l, nodeToExplore[1] + s] == 1 and not [nodeToExplore[0] + l, nodeToExplore[1]+s] in exploredNodes and not [nodeToExplore[0] + l, nodeToExplore[1]+s] in nodesToExplore):
                        nodesToExplore.append([nodeToExplore[0] + l, nodeToExplore[1] + s])
                    
                    elif(not (l == 0 and s == 0) and nodeToExplore[0] + l >= 0 and nodeToExplore[0] + l < np.size(nodeEdgeMatrix,0) and nodeToExplore[1] + s >= 0 and nodeToExplore[1] + s < np.size(nodeEdgeMatrix,1) and nodeEdgeMatrix[nodeToExplore[0] + l, nodeToExplore[1] + s] == 2 and not [nodeToExplore[0] + l, nodeToExplore[1] + s] == parentLocation):
                            return Connection(parentName, graph.getNodeByLocation(nodeToExplore[0] + l, nodeToExplore[1] + s).nodeId,1)
            iterNum = iterNum + 1    
            exploredNodes.append(nodeToExplore)
                    
    
    
        

