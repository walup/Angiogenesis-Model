from enum import Enum
import random 
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class TherapyType(Enum):
    RADIOTHERAPY = 0
    IMMUNOTHERAPY = 1
    CHEMOTHERAPY = 2
    ANGIOTHERAPY = 3


class CellType(Enum):
    PROLIFERATING = [28/255, 241/255, 93/255]
    COMPLEX = [26/255, 69/255, 245/255]
    DEAD = [245/255, 72/255, 27/255]
    NECROTIC = [130/255, 130/255, 130/255]

class GrowthPhase(Enum):
    AVASCULAR = 0
    VASCULAR = 1

class Therapy:
    
    def __init__(self, therapyType, *args):
        self.therapyType = therapyType
        self.inheritanceResistanceProbability = 0.6
        self.necrosisTherapyRate = 0.15
        #Radioterapia
        if(therapyType == TherapyType.RADIOTHERAPY):
            #El dia en que empieza la terapia siempre será el primer parámetro
            self.startDay = args[0]
            #El segundo parámetro es el factor gamma para las células que se encuentran en la fase G0
            self.g0Gamma = args[1]
            self.alpha = args[2]
            self.beta = args[3]
            self.cycleTime = args[4]
            
            self.dose = args[5]
            self.thresholdOxygen = args[6]
            self.delayTime = args[7]
            self.initMitoticProb = args[8]
            self.finalMitoticProb = args[9]
        
        #Quimioterapia
        elif(therapyType == TherapyType.CHEMOTHERAPY):
            #El primer parámetro es el día en el que empieza el tratamiento
            self.startDay = args[0]
            #El segundo parámetro es un diccionario que en función de el tipo de célula arroja la resistencia de la célula
            self.treatmentResistances = args[1]
            #El tercer parámetro será la tasa de muerte para cada tipo de célula, este también es un diccionario
            self.killRates = args[2]
            #El cuarto parámetro será el coeficiente de atenuación para cad atipo de célula
            self.attenuationCoefficients = args[3]
            #El quinto parámetro es el número de pasos que se aplicará el tratamiento 
            self.nTreatmentSteps = args[4]
            #El sexto parámetro es la constante de tiempo para el decaimiento exponencial de células
            self.tau = args[5]
            #El septimo parámetro es un factor que tiene que ver con la farmacocinética del medicamento
            #No me gusta usar mayúsculas en las variables, pero he tratado de adherirme al nombre que dan
            #a las variables en el artículo.
            self.PK = args[6]
            #Vamos a suponer que los siguientes dos parámetros son el ancho y alto de la rejilla
            self.widthAreaTreatment = args[7]
            self.heightAreaTreatment = args[8]
            #La concentración inicial de medicamento
            self.initialMedConcentration = args[9]
            self.resistanceCellsRatio = args[10]
            self.medAbsorptionCells = args[11]
            self.medDifussionConstant = args[12]
            self.medConcentration = np.zeros((self.heightAreaTreatment, self.widthAreaTreatment))
            #Supondemos que el medicamente comienza a entrar desde la frontera
            self.medConcentration[0,:] = self.initialMedConcentration
            self.medConcentration[:,0] = self.initialMedConcentration
            self.medConcentration[self.heightAreaTreatment - 1, :] = self.initialMedConcentration
            self.medConcentration[:, self.widthAreaTreatment - 1] = self.initialMedConcentration
            self.applicationSteps = args[13]
        
        elif(therapyType == TherapyType.IMMUNOTHERAPY):
            #Primer parámetro es el día en que comienza la terapia
            self.startDay = args[0]
            #Segundo parámetro es el número de pasos que se aplicará la terapia
            self.therapySteps = args[1]
            #Los valores finales de las probabilidades de transición de estados del autómata
            self.rProlifFinal = args[2]
            self.rBindingFinal = args[3]
            self.rEscapeFinal = args[4]
            self.rLysisFinal = args[5]
            self.rDecayFinal = args[6]
            #Valores iniciales de las probabilidades de transición del autómata
            self.rProlifInitial = args[7]
            self.rBindingInitial = args[8]
            self.rEscapeInitial = args[9]
            self.rLysisInitial = args[10]
            self.rDecayInitial = args[11]
            self.fixedSteprProlif = (self.rProlifFinal - self.rProlifInitial)/self.therapySteps
            self.fixedSteprBinding = (self.rBindingFinal - self.rBindingInitial)/self.therapySteps
            self.fixedSteprEscape = (self.rEscapeFinal - self.rEscapeInitial)/self.therapySteps
            self.fixedSteprLysis = (self.rLysisFinal - self.rLysisInitial)/self.therapySteps
            self.fixedSteprDecay = (self.rDecayFinal - self.rDecayInitial)/self.therapySteps
        
        elif(therapyType == TherapyType.ANGIOTHERAPY):
            self.startDay = args[0]
            self.locationsToTreat = args[1]
            self.vesselKillThreshold = args[2]
            self.potentialConstant = args[3]
            self.sourceAmplification = args[4]
            self.stepDeathProbability = args[5]
            
            #Alright, so what we are going to do is first create a vessel killing map 
            n = np.size(self.locationsToTreat, 0)
            m = np.size(self.locationsToTreat, 1)
            sourcePositions = []
            
            for i in range(0,n):
                for j in range(0,m):
                    if(self.locationsToTreat[i,j] == 1):
                        sourcePositions.append([i,j])
            
            vesselKillPotential = np.zeros((n,m))
            epsilon = 0.5
            
            for i in range(0,n):
                for j in range(0,m):
                    potential = 0
                    for s in sourcePositions:
                        dst = np.sqrt((s[0] - i)**2 + (s[1] - j)**2)
                        if(dst > epsilon):
                            potential = potential + (self.potentialConstant/dst)
                        elif(dst < epsilon):
                            potential = potential + self.potentialConstant*self.sourceAmplification
                    
                    vesselKillPotential[i,j] = potential
            
            
            maxPotential = np.max(np.max(vesselKillPotential))
            vesselKillPotential = vesselKillPotential/maxPotential
            
            self.vesselKillPotential = vesselKillPotential
            self.deathTargettedVessels = []
                            
                            
            
            
            
    def updateTherapy(self, step, cell, *args):
        if(self.therapyType == TherapyType.RADIOTHERAPY):
            cell.countCycle = cell.countCycle + 1
            #Se determina cuales de las células serán afectadas por el tratamiento 
            if(step == self.startDay):
                stageCellCycle = (cell.countCycle % self.cycleTime)//(self.cycleTime//4)
                gamma = self.g0Gamma*(1.5)**(stageCellCycle)
                oxygenConcentration = args[0]
                oer = 0
                if(oxygenConcentration > self.thresholdOxygen):
                    oer = 1
                else:
                    oer = 1 - (oxygenConcentration/self.thresholdOxygen)
            
                dOER = self.dose/oer
                probTarget = 1 - np.exp(-gamma*(self.alpha*dOER + self.beta*dOER**2))
                if(random.random() < probTarget):
                    cell.therapyAffected = True
            elif(step > self.startDay and cell.cellType == CellType.PROLIFERATING and cell.therapyAffected):
                if(step - self.startDay < self.delayTime):
                    if(random.random() < self.initMitoticProb):
                        if(random.random() < self.necrosisTherapyRate):
                            cell.turnNecrotic()
                        else:
                            cell.cellType = CellType.DEAD
                else:
                    if(random.random() < self.finalMitoticProb):
                        if(random.random() < self.necrosisTherapyRate):
                            cell.turnNecrotic()
                        else:
                            cell.cellType = CellType.DEAD
                
        
        elif(self.therapyType == TherapyType.CHEMOTHERAPY):
            cell.countCycle = cell.countCycle + 1
            #Dia en que se inicia el tratamiento
            if(step == self.startDay):
                #Ahora vamos a establecer si la célula será resistente al tratamiento 
                if(random.random() < 1 - self.resistanceCellsRatio):
                    cell.therapyAffected = True
                else:
                    cell.therapyAffected = False
            
            #Días subsecuentes al inicio del tratamiento
            if(step - self.startDay > 0 and cell.therapyAffected):
                cellType = cell.cellType
                killRate = self.killRates[cellType]
                resistance = self.treatmentResistances[cellType]
                attenuationCoefficient = self.attenuationCoefficients[cellType]
                concentration = self.medConcentration[cell.y, cell.x]
                li = (killRate*concentration)/(resistance*self.nTreatmentSteps + 1)
                #print("li"+str(li))
                probKill = li*self.PK*np.exp(-attenuationCoefficient*(step - self.startDay- self.nTreatmentSteps*self.tau))
                cellCycle = cell.countCycle%4
                #if(probKill > 0):
                    #print(killRate*concentration)
                    #print(li)
                    #print(probKill)
                if(random.random() < probKill and cellCycle == 1):
                    if(random.random() < self.necrosisTherapyRate):
                        cell.turnNecrotic()
                    else:
                        cell.cellType = CellType.DEAD
        
    
    def globalTherapyUpdate(self, step, tissue):
        if(self.therapyType == TherapyType.IMMUNOTHERAPY):
            #Día en el que inicia el tratamiento
            if(step  - self.startDay > 0 and step - self.startDay <= self.therapySteps):
                #Proliferación
                tissue.rProlif = tissue.rProlif + self.fixedSteprProlif
                tissue.rBinding = tissue.rBinding + self.fixedSteprBinding
                tissue.rEscape = tissue.rEscape + self.fixedSteprEscape 
                tissue.rLysis = tissue.rLysis + self.fixedSteprLysis
                tissue.rDecay = tissue.rDecay + self.fixedSteprDecay
        
        elif(self.therapyType == TherapyType.ANGIOTHERAPY):
            #I don't want to add a new method for initializing globally so, this will have to do 
            if(step == self.startDay and tissue.growthPhase == GrowthPhase.VASCULAR):
                n = np.size(self.vesselKillPotential, 0)
                m = np.size(self.vesselKillPotential, 1)
                vesselPositions = tissue.bloodVesselPositions
                
                for i in range(0,n):
                    for j in range(0,m):
                        if(vesselPositions[i,j] == 1 and self.vesselKillPotential[i,j] > self.vesselKillThreshold):
                            #Append all vessels that will be eliminated
                            self.deathTargettedVessels.append([i,j])
                
                
                            
                
            elif(step - self.startDay > 0):
                nVessels = len(self.deathTargettedVessels)
                if(nVessels > 0):
                    removedVessels = []
                    for i in range(0,nVessels):
                        if(random.random() < self.stepDeathProbability):
                            vesselPosition = self.deathTargettedVessels[i]
                            removedVessels.append(vesselPosition)
                            index1 = vesselPosition[0]
                            index2 = vesselPosition[1]
                            #The vessel is killed so it won't supply nutrient anymore.
                            tissue.nutrient.putValue(index1, index2, 0)
                
                    for i in range(0,len(removedVessels)):
                        self.deathTargettedVessels.remove(removedVessels[i])
                
                
            
    
    def updateTherapyDistribution(self, step, occupiedCells):
        if(self.therapyType == TherapyType.CHEMOTHERAPY):
            
            if(step - self.startDay == self.applicationSteps):
                self.medConcentration[0,:] = 0
                self.medConcentration[:,0] = 0
                self.medConcentration[self.heightAreaTreatment - 1, :] = 0
                self.medConcentration[:, self.widthAreaTreatment - 1] = 0
            
            
            if(step - self.startDay > 0):
                for i in range(1,self.heightAreaTreatment-1):
                    for j in range(1,self.widthAreaTreatment-1):
                        laPlacian = self.medConcentration[(i + 1)%self.heightAreaTreatment, j] + self.medConcentration[(i - 1)%self.heightAreaTreatment, j] + self.medConcentration[i, (j + 1)%self.widthAreaTreatment] + self.medConcentration[i, (j - 1)%self.widthAreaTreatment] - 4*self.medConcentration[i, j]
                        absorption  = 0
                        if(occupiedCells[i,j] == 1):
                            absorption = self.medAbsorptionCells
                        deltaConcentration = self.medDifussionConstant*laPlacian - absorption
                        if(self.medConcentration[i,j] + deltaConcentration >= 0):
                            self.medConcentration[i,j] = self.medConcentration[i,j] + deltaConcentration
                    
        
    def getTreatmentAffectionInhertance(self, motherTreatmentAffection):
        if(motherTreatmentAffection == False and random.random() < self.inheritanceResistanceProbability):
            return False
        else:
            return True
        
    def isStarted(self, step):
        if(self.therapyType == TherapyType.RADIOTHERAPY):
            return step >= self.startDay
        elif(self.therapyType == TherapyType.CHEMOTHERAPY):
            return step >= self.startDay

class Cell:
    
    def __init__(self, x, y, cellType, cycleTime, treatmentAffected):
        self.x = x
        self.y = y
        self.cellType = cellType
        #self.oxygenThreshold = 0.1
        self.oxygenThreshold = 0.001
        self.quiescent = False
        #Parámetros de la célula para las terapias
        #Define si la célula será afectada por el tratamiento
        self.therapyAffected = treatmentAffected
        #Lleva el ciclo de vida de la célula, lo cual es importante en tratamientos como la radioterapia
        self.countCycle = cycleTime
    
    def __eq__(self, other):
        self.x == other.x and self.y == other.y
    
    def turnNecrotic(self):
        self.cellType = CellType.NECROTIC
        
    def breathe(self, oxygenConcentration):
        if(oxygenConcentration < self.oxygenThreshold):
            self.turnNecrotic()
    
    def setQuiescent(self, quiescent):
        self.quiescent = quiescent

class Nutrient:
    
    def __init__(self, width, height, diffusionConstant, healthyCellConsumption, consumptionProlif, consumptionQuiescent):
        self.width = width
        self.height = height
        self.nutrientConcentration = np.zeros((height, width))
        self.consumptionProlif = consumptionProlif
        self.consumptionQuiescent = consumptionQuiescent
        self.diffusionConstant = diffusionConstant
        self.healthyCellConsumption = healthyCellConsumption
    
    def putValue(self, i,j, value):
        self.nutrientConcentration[i,j] = value
    
    def initializeNutrient(self):
        self.nutrientConcentration = np.zeros((self.height, self.width))
    
    def updateNutrient(self, cell, x, y, growthPhase, vesselPositions):
        index1 = y
        index2 = x
        if(not (cell is None) and growthPhase == GrowthPhase.AVASCULAR):
            if(cell.cellType  == CellType.PROLIFERATING):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionProlif
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
            elif(cell.quiescent == True):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionQuiescent
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
                #print("quiescent case")
        
        elif(cell is None and growthPhase == GrowthPhase.AVASCULAR):
            laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
            deltaConcentration = self.diffusionConstant*laPlacian - self.healthyCellConsumption
            self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
            #Do nothing
        
        elif(not (cell is None) and growthPhase == GrowthPhase.VASCULAR and vesselPositions[index1, index2] == 0):
            if(cell.cellType  == CellType.PROLIFERATING):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionProlif
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
            elif(cell.quiescent == True):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionQuiescent
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
        
        elif(cell is None and growthPhase == GrowthPhase.VASCULAR and vesselPositions[index1,index2] == 0):
            laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
            deltaConcentration = self.diffusionConstant*laPlacian - self.healthyCellConsumption
            self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
            #Do nothing
        
    def getNutrientValue(self, i,j):
        return self.nutrientConcentration[i, j]
    
class ECM:
    
    def __init__(self, width, height, ec, et):
        self.width = width
        self.height = height
        self.extraCellularMatrix = np.zeros((self.height, self.width))
        #Constante de degradación de la matriz extracelular
        self.ec = ec
        #Umbral de degradación en el que es posible invadir 
        self.et = et
        self.initializeMatrix()
    
    def initializeMatrix(self):
        for i in range(1, self.height-1):
            for j in range(1,self.width-1):
                #La ECM inicia con valores de [0.8,1.2]
                self.extraCellularMatrix[i,j] = 0.8 + random.random()*(1.2 - 0.8)
    
    def updateMatrix(self, nNeighbors, i,j):
        deltaECM = -self.ec*nNeighbors*self.extraCellularMatrix[i,j]
        self.extraCellularMatrix[i,j] = self.extraCellularMatrix[i,j] + deltaECM
    
    def canInvadePosition(self, i, j):
        if(self.extraCellularMatrix[i,j] < self.et):
            return True
        return False

class Tissue:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        #En esta matriz registraremos las posiciones ocupadas por células no necróticas
        self.occupiedPositions = np.zeros((self.height, self.width))
        #En esta matriz se registran las posiciones donde hay células necróticas
        self.necroticPositions = np.zeros((self.height, self.width))
        self.quiescentCells = np.zeros((self.height, self.width))
        self.cells = []
        self.colorNecrotic = [176/255, 176/255, 176/255]
        self.colorVessels = [201/255, 74/255, 0/255]
        #Tasa de consumo de la matriz extracelular
        self.ec = 0.1
        self.et = 0.05
        #Otros parámetros
        self.rProlif = 0.85
        self.rBinding = 0.1
        self.rEscape = 0.5
        self.rLysis = 0.35
        self.rDecay = 0.35
        self.K = 1000
        
        #Parámetros del nutriente
        self.difussionConstant = 0.01
        self.consumptionProlif = 0.01
        self.consumptionQuiescent = 0.005
        self.consumptionHealthy = 0.0001
        
        #Inicializamos la matriz extracelular y el nutriente
        self.ecm = ECM(self.width, self.height, self.ec, self.et)
        self.nutrient = Nutrient(self.width, self.height, self.difussionConstant, self.consumptionHealthy, self.consumptionProlif, self.consumptionQuiescent)
        self.initializeNutrientAndECM()
        self.growthPhase = GrowthPhase.AVASCULAR
        
        
        #Parámetros extra para las terapias
        self.cellCycleTime = 4
        self.therapies = []
        
    def initializeNutrientAndECM(self):
        self.ecm.initializeMatrix()
        self.nutrient.initializeNutrient()
        #Voy  a probar poner un valor de nutriente al centro a ver si funciona
        for i in range(0,self.height):
            for j in range(0,self.width):
                self.nutrient.putValue(i,j,1)
        
        self.nutrient.nutrientConcentration[0,:] = 2
        self.nutrient.nutrientConcentration[:,0] = 2
        self.nutrient.nutrientConcentration[self.height-1,:] = 2
        self.nutrient.nutrientConcentration[:,self.width-1] = 2
    
    
    def initializeNutrientWithBloodVessels(self, bloodVesselPositions):
        self.bloodVesselPositions = bloodVesselPositions
        self.growthPhase = GrowthPhase.VASCULAR
        for i in range(0,np.size(bloodVesselPositions, 0)):
            for j in range(0, np.size(bloodVesselPositions,1)):
                if(bloodVesselPositions[i,j] == 1):
                    self.nutrient.putValue(i,j,2)
                
                else:
                    self.nutrient.putValue(i,j,0.5)
        
        self.nutrient.nutrientConcentration[0,:] = 0
        self.nutrient.nutrientConcentration[:,0] = 0
        self.nutrient.nutrientConcentration[self.height - 1, :] = 0
        self.nutrient.nutrientConcentration[:,self.width-1] = 0
    
    def initializeChemoWithBloodVessels(self, bloodVesselPositions):
        for i in range(0,len(self.therapies)):
            therapy = self.therapies[i]
            if(therapy.therapyType == TherapyType.CHEMOTHERAPY):
                for j in range(0,np.size(therapy.medConcentration,0)):
                    for l in range(0,np.size(therapy.medConcentration,1)):
                        if(bloodVesselPositions[j,l] == 1):
                            self.therapies[i].medConcentration[j,l] = therapy.initialMedConcentration
                        
                        else:
                            self.therapies[i].medConcentration[j,l] = 0
        
    #Cuenta el número de vecinos alrededor de la posición x, y donde x y y son enteros
    def countNeighbors(self, x, y):        
        sumNeighbors = 0
        for i in range(-1,2):
            for j in range(-1,2):
                if(i != 0 or j!= 0):
                    sumNeighbors = sumNeighbors + self.occupiedPositions[(y + j)%self.width, (x + i)%self.height]
        return sumNeighbors
    
    #Obtiene posiciones disponibles para invadir por el tumor
    def getPositionsToInfest(self, x, y):
        positions = [];
        for i in range(-1,2):
            for j in range(-1, 2):
                if(i != 0 or j!= 0):
                    row = (y + i)%self.width
                    col = (x + j)%self.height
                    if(self.occupiedPositions[row, col]  == 0 and self.necroticPositions[row, col] == 0 and self.ecm.canInvadePosition(row, col)):
                        positions.append([row, col])
        
        return positions
    
    #Actualiza el nutriente y la matriz extracelular
    def updateNutrientAndECM(self, growthPhase):
        for i in range(0,self.height):
            for j in range(0,self.width):
                nNeighbors = self.countNeighbors(j,i)
                self.ecm.updateMatrix(nNeighbors, i, j)
                #Update nutrient concentration of healthy cells
                if(self.occupiedPositions[i,j] == 0):
                    if(growthPhase == GrowthPhase.AVASCULAR):
                        self.nutrient.updateNutrient(None, j,i, growthPhase, None)
                    elif(growthPhase == GrowthPhase.VASCULAR):
                        self.nutrient.updateNutrient(None, j, i, growthPhase, self.bloodVesselPositions)
        #Update nutrient concentration of cancer and other types of cells
        for i in range(0,len(self.cells)):
            if(growthPhase == GrowthPhase.AVASCULAR):
                self.nutrient.updateNutrient(self.cells[i], self.cells[i].x, self.cells[i].y, growthPhase, None)
            
            elif(growthPhase == GrowthPhase.VASCULAR):
                self.nutrient.updateNutrient(self.cells[i], self.cells[i].x, self.cells[i].y, growthPhase, self.bloodVesselPositions)
        
        
        
    def updateCells(self, step):
        cellsToDelete = []
        #Actualizamos aleatoriamente las células
        indsList = list(range(0,len(self.cells)))
        random.shuffle(indsList)
        for i in indsList:
            cell = self.cells[i]
            r = random.random()
            #Proliferadoras
            if(cell.quiescent == True and len(self.getPositionsToInfest(cell.x, cell.y)) > 0):
                cell.setQuiescent(False)
                self.quiescentCells[cell.y,cell.x] = 0
            
            oxygenConcentration = self.nutrient.getNutrientValue(cell.y, cell.x)
            self.cells[i].breathe(oxygenConcentration)
            
            if(cell.cellType == CellType.PROLIFERATING):
                if(r <= self.rProlifPrime):
                    normalCells = self.getPositionsToInfest(cell.x, cell.y)
                    #print(len(normalCells))
                    if(len(normalCells)>0):
                        normalCell = random.choice(normalCells)
                        therapyResistance = self.getTreatmentResistance(step, cell)
                        self.addProliferatingCell(normalCell[1], normalCell[0], therapyResistance, step)
                    else:
                        cell.setQuiescent(True)
                        self.quiescentCells[cell.y, cell.x] = 1
                        
                else:
                    if(r <= 1 - self.rBinding):
                        self.cells[i].cellType = CellType.COMPLEX
                self.updateTherapy(step, cell, oxygenConcentration)
            #Complejas
            elif(cell.cellType == CellType.COMPLEX):
                if(r <= self.rEscape):
                    self.cells[i].cellType = CellType.PROLIFERATING
                elif(r >= 1 - self.rLysis):
                    self.cells[i].cellType = CellType.DEAD
                    
            elif(cell.cellType == CellType.DEAD):
                if(r < self.rDecay):
                    cellsToDelete.append(cell)
            
            if(self.cells[i].cellType == CellType.NECROTIC):
                self.necroticPositions[cell.y, cell.x] = 1
                cellsToDelete.append(cell)
        
        for i in range(0,len(cellsToDelete)):
            if(cellsToDelete[i] in self.cells):
                self.removeCell(cellsToDelete[i])
    
    
    def getCellCounts(self):
        
        proliferatingCells = 0
        complexCells = 0
        necroticCells = 0
        for i in range(0,len(self.cells)):
            cell = self.cells[i]
            if(cell.cellType == CellType.PROLIFERATING):
                proliferatingCells = proliferatingCells + 1
            elif(cell.cellType == CellType.COMPLEX):
                complexCells = complexCells + 1
        necroticCells = sum(sum(self.necroticPositions))
        return [proliferatingCells, complexCells, len(self.cells), necroticCells]
    
    def plotEvolution(self,ax):
        n = np.size(self.cellCountSeries, 0)
        ax.plot(self.cellCountSeries[:,0], color = CellType.PROLIFERATING.value, label = "Proliferating", linewidth = 2)
        ax.plot(self.cellCountSeries[:,1], color = CellType.COMPLEX.value, label = "Complex", linewidth = 2)
        ax.plot(self.cellCountSeries[:,3], color = CellType.NECROTIC.value, label = "Necrotic", linewidth = 2)
        ax.plot(self.cellCountSeries[:, 2], color = "#ff9a26", label = "All Cells", linewidth = 2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of cells")
        ax.legend()
    
    def evolveTissueInitially(self, nSteps):
        self.cellCountSeries = np.zeros((nSteps+1, 3)) 
        counts = self.getCellCounts()
        self.cellCountSeries[0,:] = counts
        for i in tqdm(range(1, nSteps+1)):
            #self.rProlifPrime = self.rProlif
            self.rProlifPrime = self.rProlif*(1 - counts[0]/self.K)
            #Actualizamos el nutriente y la matriz extracelular
            self.updateNutrientAndECM(self.growthPhase)
            #Actualizamos las células
            self.updateCells(i)
            #Actualizamos la distribución del tratamiento (fármaco) si es necesario
            self.updateTherapyDistribution(i)
            self.updateTherapyGlobally(i, self)
            #Guardamos una snapshot del tumor para poder armar una animación 
            counts = self.getCellCounts()
            self.cellCountSeries[i,:] = counts
    
    def continueTissueEvolution(self, nSteps):
        oldCounts = self.cellCountSeries
        nOldCounts = np.size(oldCounts,0)
        self.cellCountSeries = np.zeros((nOldCounts + nSteps, 3))
        self.cellCountSeries[0:nOldCounts,:] = oldCounts
        counts = self.getCellCounts()
        for i in tqdm(0,nSteps):
            stepNum = i + nOldCounts
            self.rProlifPrime = self.rProlif*(1 - counts[0]/self.K)
            self.updateNutrientAndECM(self.growthPhase)
            self.updateCells(stepNum)
            self.updateTherapyDistribution(stepNum)
            self.updateTherapyGlobally(stepNum, self)
            counts = self.getCellCounts()
            self.cellCountSeries[stepNum,:] = counts
        
    def continueTissueEvolutionWithMovie(self, nSteps, includeNecrotic):
        oldEvolutionMovie = self.evolutionMovie
        oldNFrames = np.size(oldEvolutionMovie, 3)
        self.evolutionMovie = np.zeros((self.height, self.width, 3, oldNFrames + nSteps))
        self.evolutionMovie[:,:,:,0:oldNFrames] = oldEvolutionMovie
        oldCounts = self.cellCountSeries
        nOldCounts = np.size(oldCounts,0)
        self.cellCountSeries = np.zeros((nOldCounts + nSteps,4))
        self.cellCountSeries[0:nOldCounts,:] = oldCounts
        counts = self.getCellCounts()
        for i in tqdm(range(0,nSteps)):
            stepNum = i + nOldCounts
            self.rProlifPrime = self.rProlif*(1 - counts[0]/self.K)
            self.updateNutrientAndECM(self.growthPhase)
            self.updateCells(stepNum)
            self.updateTherapyDistribution(stepNum)
            self.updateTherapyGlobally(stepNum, self)
            counts = self.getCellCounts()
            self.cellCountSeries[stepNum,:] = counts
            self.evolutionMovie[:,:,:,stepNum] = self.getPicture(includeNecrotic)
    
    
    def evolveWithMovieInitially(self, nSteps, includeNecrotic):
        self.evolutionMovie = np.zeros((self.height, self.width, 3, nSteps + 1))
        self.cellCountSeries = np.zeros((nSteps+1, 4)) 
        counts = self.getCellCounts()
        self.cellCountSeries[0,:] = counts
        self.evolutionMovie[:,:,:,0] = self.getPicture(includeNecrotic)
        for i in tqdm(range(1, nSteps+1)):
            #self.rProlifPrime = self.rProlif
            self.rProlifPrime = self.rProlif*(1 - counts[0]/self.K)
            #Actualizamos el nutriente y la matriz extracelular
            self.updateNutrientAndECM(self.growthPhase)
            #Actualizamos las células
            self.updateCells(i)
            #Actualizamos la distribución del tratamiento (fármaco) si es necesario
            self.updateTherapyDistribution(i)
            self.updateTherapyGlobally(i, self)
            #Vamos guardando las series de tiempo de los distintos tipos de célula
            counts = self.getCellCounts()
            self.cellCountSeries[i,:] = counts
            #Guardamos una snapshot del tumor para poder armar una animación 
            self.evolutionMovie[:,:,:,i] = self.getPicture(includeNecrotic)
    
    
    def evolveWithMovieMultipleTreatments(self, nSteps, includeNecrotic, treatments, treatmentDates, growthPhase):
        self.evolutionMovie = np.zeros((self.height, self.width, 3, nSteps + 1))
        self.cellCountSeries = np.zeros((nSteps + 1, 4))
        counts = self.getCellCounts()
        self.cellCountSeries[0,:] = counts
        self.evolutionMovie[:,:,:,0] = self.getPicture(includeNecrotic)
        treatmentsApplied = np.zeros(len(treatmentDates))
        for i in tqdm(range(1,nSteps + 1)):
            #Verify if we need to add some treatment
            for j in range(0,len(treatmentDates)):
                if(treatmentsApplied[j] == 0 and treatmentDates[j] == i):
                    self.addTherapy(treatments[j])
                    treatmentsApplied[j] = 1
                    print("Added new therapy")
                    
            self.rProlifPrime = self.rProlif*(1 - counts[0]/self.K)
            self.updateNutrientAndECM(self.growthPhase)
            self.updateCells(i)
            self.updateTherapyDistribution(i)
            self.updateTherapyGlobally(i,self)
            counts = self.getCellCounts()
            self.cellCountSeries[i,:] = counts
            self.evolutionMovie[:,:,:,i] = self.getPicture(includeNecrotic)
            
                
    
    def addProliferatingCell(self, x, y, treatmentAffected, step):
        self.cells.append(Cell(x,y,CellType.PROLIFERATING,  step, treatmentAffected))
        self.occupiedPositions[y,x] = 1       
        
    def getPicture(self, includeNecrotic):
        picture = np.zeros((self.height, self.width, 3))
        antiangiogenicPicture = (len(self.therapies) > 0 and any([t.therapyType == TherapyType.ANGIOTHERAPY for t in self.therapies]))
        
        
        if(self.growthPhase == GrowthPhase.VASCULAR and not antiangiogenicPicture):
            for i in range(0,np.size(self.bloodVesselPositions,0)):
                for j in range(0,np.size(self.bloodVesselPositions,1)):
                    if(self.bloodVesselPositions[i,j] == 1):
                        picture[i,j,:] = self.colorVessels
        
        elif(self.growthPhase == GrowthPhase.VASCULAR and antiangiogenicPicture):
            nutrientConcentrations = self.nutrient.nutrientConcentration
            for i in range(0,np.size(nutrientConcentrations,0)):
                for j in range(0,np.size(nutrientConcentrations,1)):
                    if(nutrientConcentrations[i,j] == 2):
                        picture[i,j,:] = self.colorVessels
                        
                        
        
        for i in range(0, len(self.cells)):
            picture[self.cells[i].y, self.cells[i].x,:] = self.cells[i].cellType.value
        if(includeNecrotic):
            for i in range(0, self.height):
                for j in range(0, self.width):
                    if(self.necroticPositions[i,j] == 1):
                        picture[i,j,:] = self.colorNecrotic
            
        
        return picture
            
    def removeCell(self, cell):
        self.cells.remove(cell)
        self.occupiedPositions[cell.y, cell.x] = 0
        
    def addTherapy(self, therapy):
        self.therapies.append(therapy)
        
    #Funciones de inicialización y aplicación de tratamiento
    def updateTherapy(self, step, cell, *args):
        if(len(self.therapies) != 0):
            for i in range(0,len(self.therapies)):
                self.therapies[i].updateTherapy(step,cell,*args)
    
    def getTreatmentResistance(self, step, cell):
        if(len(self.therapies) != 0 and self.therapies[0].isStarted(step)):
            return self.therapies[0].getTreatmentAffectionInhertance(cell.therapyAffected)
        return False
    
    def updateTherapyDistribution(self, step):
        if(len(self.therapies) != 0):
            for i in range(0,len(self.therapies)):
                self.therapies[i].updateTherapyDistribution(step, self.occupiedPositions)
    
    def updateTherapyGlobally(self,step, *args):
        if(len(self.therapies) != 0):
            for i in range(0,len(self.therapies)):
                self.therapies[i].globalTherapyUpdate(step,*args)
                
    
    def exportTumorInstance(self, fileName):
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
            print("Object saved to "+fileName)
    
    def importTumorInstance(self, fileName):
        tissueInstance = pickle.load(open(fileName, 'rb'))
        return tissueInstance