import math
import random
import re
import time
import tkinter
from threading import Thread
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import *

from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt

MAX_ELITISM_RATE = 0
MAX_TOURNAMENT_RATE = 0
TOURNAMENT_BASE_STAGE = 8
MAX_MULTIPOINT_POINTS = 4
MUTATION_RATE = 0
ONE_POINT_CROSSOVER_RATE = 0
MULTIPOINT_CROSSOVER_RATE = 0
ROOT = None
CANVAS = None
PLOT = None
TEXT_BOX = None
AVERAGE_FITNESS_LIST = []
BEST_FITNESS_LIST = []
GENERATIONS = []

class City:
    name = None
    x = 0
    y = 0

    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

class Offspring:
    chromosome = None
    fitness = 0
    distance = 0

    def calculateDistance(self, firstCity, secondCity):
        d = math.pow((secondCity.x - firstCity.x), 2.0) + math.pow((secondCity.y - firstCity.y), 2.0)
        return math.sqrt(d)

    def createFitness(self):
        for i in range(0, len(self.chromosome) - 1):
            self.distance += self.calculateDistance(self.chromosome[i], self.chromosome[i+1])
        self.fitness = 1 / self.distance

    def print(self):
        output = ""
        for allele in self.chromosome:
            output += allele.name + " "
        return output[:len(output)-1]

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.createFitness()

class Generation:
    previousGeneration = None
    bestOffspring = None

    def random(self):
        if(random.randint(0, 99) < 50):
            return True
        return False

    def rouletteWheel(self):
        S = sum(offspring[0].fitness for offspring in self.previousGeneration)
        rand = random.uniform(0, S)
        P = 0
        for population in self.previousGeneration:
            P += population[0].fitness
            if(P >= rand):
                return population
        return None

    def elitismSelection(self):
        topOffspringsList = sorted(self.previousGeneration,key=lambda x: x[0].fitness, reverse=True)
        for i in range(0, self.elitism):
            topOffspringsList[i][1] = True
            self.population[i] = topOffspringsList[i]


    def bracketWinner(self, firstContestant, secondContestant):
        if(firstContestant[0].fitness > secondContestant[0].fitness):
            return firstContestant
        return secondContestant

    def tournamentSelection(self):
        if(self.tournament % TOURNAMENT_BASE_STAGE != 0):
            diff = (TOURNAMENT_BASE_STAGE - (self.tournament % TOURNAMENT_BASE_STAGE))
            self.tournament += diff
            self.roulette = self.roulette - diff

        tournamentMembers = 0
        while(tournamentMembers < self.tournament):
            i = 0
            tournamentBaseStage = []
            while(i < TOURNAMENT_BASE_STAGE):#pick 8 participants
                index = random.randint(0, len(self.previousGeneration) - 1)
                if(self.random() and not self.previousGeneration[index][1]):
                    tournamentBaseStage.append(self.previousGeneration[index])
                    i += 1
            while(len(tournamentBaseStage) != 1):
                tournamentBaseStage = [tournamentBaseStage[j:j + 2] for j in range(0, len(tournamentBaseStage), 2)] #4x2 brackets
                winners = []
                for bracket in tournamentBaseStage:
                    winners.append(self.bracketWinner(bracket[0], bracket[1]))
                tournamentBaseStage = winners
            ix = self.previousGeneration.index(tournamentBaseStage[0])
            #self.previousGeneration[ix][1] = True
            if(self.elitism + tournamentMembers <= len(self.population)-1):
                self.population[self.elitism + tournamentMembers] = self.previousGeneration[ix]
            tournamentMembers += 1

    def createParentPairs(self):
        self.parentPairs = []
        index = 0
        while(index < len(self.population) - 1):
            parents = []
            if(self.population[index] != self.population[index+1]):
                parents.append(self.population[index])
                parents.append(self.population[index+1])
            else:
                parents.append(self.population[index])
                for i in range(int(len(self.population)/2), len(self.population)):
                    if(index != i and self.population[index] != self.population[i]):
                        parents.append(self.population[i])
                        break

            index += len(parents)
            self.parentPairs.append(parents)


    def offspringResolver(self, offspring, firstParent, secondParent):
        for i in range(0, len(offspring)):
            if (offspring[i] == 0):
                additionalCity = None
                for city in firstParent[0].chromosome:
                    if (not city in offspring):
                        offspring[i] = city
                        additionalCity = offspring[i]
                        break
                if (additionalCity is None):
                    for city in secondParent[0].chromosome:
                        if (not city in offspring):
                            offspring[i] = city
                            additionalCity = offspring[i]
                            break
  
    def onePointCrossover(self, firstParent, secondParent):
        if(len(firstParent[0].chromosome) != len(secondParent[0].chromosome)):
            return None, None
        start = random.randint(0, len(firstParent[0].chromosome) - 2)
        end = random.randint(start, len(firstParent[0].chromosome) - 2)

        firstOffspring = [0] * (len(firstParent[0].chromosome) - 1)
        for index in range(0, len(firstOffspring)):
            if(index >= start and index <= end):
                if(firstOffspring[index] == 0):
                    firstOffspring[index] = firstParent[0].chromosome[index]

        for i in range(0, len(firstOffspring)):
            if(firstOffspring[i] == 0):
                for city in secondParent[0].chromosome:
                    if(not city in firstOffspring):
                        firstOffspring[i] = city
                        break
        firstOffspring.append(firstOffspring[0])
        firstOffspring = Offspring(firstOffspring)

        secondOffspring = [0] * (len(secondParent[0].chromosome) - 1)
        for index in range(0, len(secondOffspring)):
            if (index >= start and index <= end):
                if (secondOffspring[index] == 0):
                    secondOffspring[index] = secondParent[0].chromosome[index]

        for i in range(0, len(secondOffspring)):
            if (secondOffspring[i] == 0):
                for city in firstParent[0].chromosome:
                    if (not city in secondOffspring):
                        secondOffspring[i] = city
                        break
        secondOffspring.append(secondOffspring[0])
        secondOffspring = Offspring(secondOffspring)

        if(firstOffspring != secondOffspring):
            return firstOffspring, secondOffspring

    def multiPointCrossoverHelper(self, offspring, breakPoints, first, second):
        j = 0
        prev = 0
        while (j < len(breakPoints)):
            if (j % 2 == 0):
                index = prev
                while (index <= breakPoints[j]):
                    if (first[0].chromosome[index] not in offspring):
                        offspring[index] = first[0].chromosome[index]
                    else:
                        for k in range(0, len(first[0].chromosome) - 1):
                            if (first[0].chromosome[k] not in offspring):
                                offspring[index] = first[0].chromosome[k]
                                break
                    index += 1
            else:
                index = prev
                while (index <= breakPoints[j]):
                    if (second[0].chromosome[index] not in offspring):
                        offspring[index] = second[0].chromosome[index]
                    else:
                        for k in range(0, len(second[0].chromosome) - 1):
                            if (second[0].chromosome[k] not in offspring):
                                offspring[index] = second[0].chromosome[k]
                                break
                    index += 1
            prev = breakPoints[j]
            j += 1
        for index in range(breakPoints[-1] + 1, len(first[0].chromosome) - 1):
            if(len(breakPoints) % 2 == 0):
                #take from first parent
                for city in first[0].chromosome:
                    if(not city in offspring):
                        offspring[index] = city
                        break
            else:
                for city in second[0].chromosome:
                    if(not city in offspring):
                        offspring[index] = city
                        break

    def multiPointCrossover(self, firstParent, secondParent):
        breakPoints = []
        previous = -1
        for i in range(0, MAX_MULTIPOINT_POINTS):
            if(previous + 1 >= len(firstParent[0].chromosome) - 1):
                break
            breakIndex = random.randint(previous + 1, len(firstParent[0].chromosome) - 2)
            breakPoints.append(breakIndex)
            previous = breakIndex

        firstOffspring = [0] * (len(firstParent[0].chromosome) - 1)
        self.multiPointCrossoverHelper(firstOffspring, breakPoints, firstParent, secondParent)
        firstOffspring.append(firstOffspring[0])
        firstOffspring = Offspring(firstOffspring)

        secondOffspring = [0] * (len(firstParent[0].chromosome) - 1)
        self.offspringResolver(secondOffspring, firstParent, secondParent)
        secondOffspring.append(secondOffspring[0])
        secondOffspring = Offspring(secondOffspring)

        if(firstOffspring != secondOffspring):
            return firstOffspring, secondOffspring


    def uniformCrossover(self, firstParent, secondParent):
        firstOffspring = [0] * len(firstParent[0].chromosome)
        for i in range(0, len(firstParent[0].chromosome)):
            if(self.random() and not firstParent[0].chromosome[i] in firstOffspring):
                firstOffspring[i] = firstParent[0].chromosome[i]
            else:
                for city in secondParent[0].chromosome:
                    if(not city in firstOffspring):
                        firstOffspring[i] = city
                        break

        firstOffspring[-1] = firstOffspring[0]
        firstOffspring = Offspring(firstOffspring)

        secondOffspring = [0] * len(secondParent[0].chromosome)
        for i in range(0, len(secondParent[0].chromosome)):
            if (self.random() and not secondParent[0].chromosome[i] in secondOffspring):
                secondOffspring[i] = secondParent[0].chromosome[i]
            else:
                for city in firstParent[0].chromosome:
                    if (not city in secondOffspring):
                        secondOffspring[i] = city
                        break

        secondOffspring[-1] = secondOffspring[0]
        secondOffspring = Offspring(secondOffspring)

        if(firstOffspring != secondOffspring):
            return firstOffspring, secondOffspring


    def crossover(self):
        offspringList = [0] * len(self.population)
        index = 0
        for parentPair in self.parentPairs:
            rnd = random.randint(0, 100)
            firstOffspring = None
            secondOffspring = None
            if(rnd < ONE_POINT_CROSSOVER_RATE):
                firstOffspring, secondOffspring = self.onePointCrossover(parentPair[0], parentPair[1])
            elif(rnd >= ONE_POINT_CROSSOVER_RATE and rnd < (ONE_POINT_CROSSOVER_RATE + MULTIPOINT_CROSSOVER_RATE)):
                firstOffspring, secondOffspring = self.multiPointCrossover(parentPair[0], parentPair[1])
            else:
                firstOffspring, secondOffspring = self.uniformCrossover(parentPair[0], parentPair[1])

            if(firstOffspring is not None or secondOffspring is not None):
                offspringList[index] = [firstOffspring, False]
                offspringList[index+1] = [secondOffspring, False]
                index += len(parentPair)

        self.population = offspringList

    def swap(self, firstCity, secondCity):
        temp = firstCity
        firstCity = secondCity
        secondCity = temp

    def mutation(self):
        for offspring in self.population:
            for i in range(0, len(offspring[0].chromosome)):
                chance = random.randint(0, 100)
                if(chance <= MUTATION_RATE):
                    randomIndex = random.randint(i, len(offspring[0].chromosome) - 1)
                    self.swap(offspring[0].chromosome[i], offspring[0].chromosome[randomIndex])

        for offspring in self.population:
            if(offspring[0].chromosome[0] != offspring[0].chromosome[-1]): #correction
                if(self.random()):
                    index = [j for j, value in enumerate(offspring[0].chromosome) if value == offspring[0].chromosome[0]][1]
                    self.swap(offspring[0].chromosome[index], offspring[0].chromosome[-1])
                else:
                    index = [j for j, value in enumerate(offspring[-1].chromosome) if value == offspring[0].chromosome[-1]][1]
                    self.swap(offspring[0].chromosome[index], offspring[0].chromosome[0])

    def __init__(self, generation):
        self.previousGeneration = generation
        self.generationAmount = len(self.previousGeneration)
        self.elitism = int((MAX_ELITISM_RATE * self.generationAmount) / 100)
        self.tournament = int((MAX_TOURNAMENT_RATE * self.generationAmount) / 100)
        self.roulette = int(((100 - (MAX_ELITISM_RATE + MAX_TOURNAMENT_RATE)) * self.generationAmount) / 100)
        self.population = [0] * self.generationAmount
        self.elitismSelection()
        self.tournamentSelection()
        l = 0
        for i in range(0, len(self.population)):
            if(not isinstance(self.population[i], int)):
                l += 1

        while(l < len(self.population)):
            offspring = self.rouletteWheel()
            if(offspring):
                self.population[l] = offspring
                l += 1

        for offspring in self.population:
            if(offspring[1]):
                offspring[1] = False

        if(self.population != self.previousGeneration):
            self.createParentPairs()
            self.crossover()
            self.mutation()
            topList = sorted(self.population,key=lambda x: x[0].fitness, reverse=True)
            self.averageFitness = sum(offspring[0].fitness for offspring in self.population) / self.generationAmount
            self.bestOffspring = topList[0][0]
            for offspringList in self.population:
                if(offspringList[0] == 0):
                    print("Wrong value detected!!!")

class TSP:
    cityList = None
    populationSize = 0
    initialPopulation = None
    bestMinimal = 0
    localMinimal = 0

    def __createInitialPopulation(self):
        for i in range(0, self.populationSize):
            tempCityList = self.cityList.copy()
            random.shuffle(tempCityList)
            tempCityList.append(tempCityList[0])
            self.initialPopulation[i] = [Offspring(tempCityList), False]

    def loopUntil(self, value, n):
        if(len(self.bestGenerationList) < n):
            return True
        sumofLastN = sum(offspring.fitness for offspring in self.bestGenerationList[-n:])
        averageSum = sumofLastN / n
        if(value.fitness >= (0.995 * averageSum) and value.fitness <= (1.005 * averageSum)):
            return False
        return True

    def shuffle(self, value, n):
        sumofLastN = sum(offspring.fitness for offspring in self.bestGenerationList[-n:])
        if(round(value.fitness, 10) == round((sumofLastN / n), 10)):
            return True
        return False

    def swap(self, first, second):
        temp = first
        first = second
        second = temp

    def __init__(self, cityList, populationSize, maxGenerations):
        self.bestGenerationList = []
        self.cityList = cityList
        self.populationSize = populationSize#pow(len(self.cityList), 2) if len(self.cityList) % 2 == 0 else pow(len(self.cityList), 2) + 1
        self.maxGenerations = maxGenerations
        self.initialPopulation = [0] * self.populationSize
        self.__createInitialPopulation()
        minimalList = sorted(self.initialPopulation,key=lambda x: x[0].fitness, reverse=True)
        self.localMinimal = minimalList[0][0].fitness
        average = sum(offspring[0].fitness for offspring in self.initialPopulation) / self.populationSize
        gen_number = 0
        population = self.initialPopulation
        self.bestGenerationList.append(minimalList[0][0])
        displayCanvas(minimalList[0][0], gen_number)
        AVERAGE_FITNESS_LIST.append(round(average, 8))
        BEST_FITNESS_LIST.append(round(self.localMinimal, 8))
        output = "[GENERATION " + str(gen_number) + "]" + " Average fitness: " + str(average) + '\n'
        output += "Best fitness: " + str(self.localMinimal) + " [DISTANCE] " + str(self.bestGenerationList[-1].distance) + '\n'
        output += "[PATH] " + self.bestGenerationList[-1].print() + '\n'
        output += "*************************************************************\n"
        TEXT_BOX.insert(END, output)
        while gen_number < maxGenerations: #self.loopUntil(self.bestGenerationList[-1], 100):
            self.bestMinimal = self.localMinimal
            generation = Generation(population)
            gen_number += 1
            output = "[GENERATION " + str(gen_number) + "]" + " Average fitness: " + str(generation.averageFitness) + '\n'
            output += "Best fitness: " + str(generation.bestOffspring.fitness) + " [DISTANCE] " + str(generation.bestOffspring.distance) + '\n'
            output += "[PATH] " + generation.bestOffspring.print() + '\n'
            output += "*************************************************************\n"
            TEXT_BOX.insert(END, output)
            displayCanvas(generation.bestOffspring, gen_number)
            #showGraphData(generation.averageFitness, generation.bestOffspring.fitness, gen_number)
            AVERAGE_FITNESS_LIST.append(round(generation.averageFitness, 8))
            BEST_FITNESS_LIST.append(round(generation.bestOffspring.fitness, 8))
            if(self.shuffle(self.bestGenerationList[-1], int(maxGenerations/10))):
                for i in range(0, len(generation.population)):
                    generation.population[i][0].chromosome = random.sample(generation.population[i][0].chromosome[:-1], len(generation.population[i][0].chromosome) - 1)
                    generation.population[i][0].chromosome.append(generation.population[i][0].chromosome[0])


            population = generation.population
            self.localMinimal = generation.bestOffspring.fitness
            self.bestGenerationList.append(generation.bestOffspring)

        #hladanie naj clena
        best_distance = self.bestGenerationList[0].distance
        best_index = 0
        for i in range(0, len(self.bestGenerationList)):
            if(self.bestGenerationList[i].distance < best_distance):
                best_distance = self.bestGenerationList[i].distance
                best_index = i

        output = "\n[BEST RESULT was found in generation " + str(best_index) + "]\n"
        output += "[DISTANCE] " + str(self.bestGenerationList[best_index].distance) + " (fitness: " + str(self.bestGenerationList[best_index].fitness) + ")\n"
        output += "[PATH] " + self.bestGenerationList[best_index].print() + "\n"
        print(output)
        TEXT_BOX.insert(END, output)
        writeToFile(str(len(cityList)) + "_" + str(populationSize) + "_" + str(maxGenerations))
            #population = generation.getpopulation
        #firstGeneration = Generation(self.initialPopulation)


def displayCanvas(offspring, generation_number):
    CANVAS.delete("all")
    genCanvas = "Generation: " + str(generation_number)
    distCanvas = "Distance: " + str(offspring.distance)
    CANVAS.create_text(55, 10, fill="black", font="Times 12 italic bold",text=genCanvas)
    CANVAS.create_text(105, 30, fill="black", font="Times 12 italic bold",text=(distCanvas))
    pathCanvas = "Path: " + str(offspring.print())
    CANVAS.create_text(int(len(pathCanvas) * 3.35), 50, fill="black", font="Times 12 italic bold",text=pathCanvas)
    for i in range(0, len(offspring.chromosome)-1):
        CANVAS.create_oval(offspring.chromosome[i].x+5, offspring.chromosome[i].y+100, offspring.chromosome[i].x+5, offspring.chromosome[i].y+100, fill="black", width=5)
        CANVAS.create_line(offspring.chromosome[i].x+5, offspring.chromosome[i].y+100, offspring.chromosome[i+1].x+5, offspring.chromosome[i+1].y+100, fill="red", width=3, arrow=LAST)
        cityName = offspring.chromosome[i].name
        fillCityName = "darkblue"
        if(not i):
            cityName = offspring.chromosome[i].name + "-START"
            fillCityName = "green"

        CANVAS.create_text(offspring.chromosome[i].x+5, offspring.chromosome[i].y+100+10, fill=fillCityName, font="Times 10 bold",text=cityName)

def writeToFile(fileName):
    f = open(fileName + ".txt", "w")
    f.write(TEXT_BOX.get(1.0, END))
    print("Output written to file")
    f.close()
def showGraphData(i):
    PLOT.clear()
    PLOT.plot(AVERAGE_FITNESS_LIST, label="Average fitness", color="blue")
    PLOT.plot(BEST_FITNESS_LIST, label="Best fitness", color="red")
    PLOT.set_xlabel('Generations')
    PLOT.set_ylabel('Fitness')
    PLOT.legend()


def start_tsp(populationSize, generations):
    tsp = TSP(CITY_LIST, populationSize, generations)

def start_program(populationSize, generations):
    t1 = Thread(target=start_tsp, args=(populationSize, generations))
    t1.start()

def init(auto, maxcities, populationSize, generations, sliderPack):
    global MAX_ELITISM_RATE
    global MAX_TOURNAMENT_RATE
    global MUTATION_RATE
    global ONE_POINT_CROSSOVER_RATE
    global MULTIPOINT_CROSSOVER_RATE
    global CITY_LIST
    if(sliderPack[0] + sliderPack[1] <= 100 and sliderPack[3] + sliderPack[4] <= 100):
        MAX_ELITISM_RATE = sliderPack[0]
        MAX_TOURNAMENT_RATE = sliderPack[1]
        MUTATION_RATE = sliderPack[2]
        ONE_POINT_CROSSOVER_RATE = sliderPack[3]
        MULTIPOINT_CROSSOVER_RATE = sliderPack[4]
        if(auto):


            if (len(generations) > 0):
                generations = int(generations)
            else:
                generations = random.randint(100, 10000)

            if(len(maxcities) > 0):
                maxcities = int(maxcities)
            else:
                maxcities = random.randint(20, 50)

            if (len(populationSize) > 0):
                populationSize = int(populationSize)
            else:
                populationSize = pow(maxcities, 2) if maxcities % 2 == 0 else pow(maxcities, 2) + 1

            if(CITY_LIST is None):
                CITY_LIST = []
                for i in range(0, maxcities):
                    generatedCity = City(str(i), random.randint(0, 700), random.randint(0, 400))
                    while(generatedCity in CITY_LIST):
                        generatedCity = City(str(i), random.randint(0, 700), random.randint(0, 400))
                    CITY_LIST.append(generatedCity)

        else:
            if(len(populationSize) > 0 and len(generations) > 0):
                populationSize = int(populationSize)
                generations = int(generations)
            else:
                showerror("Error", "Invalid settings detected")
                return
        if(len(CITY_LIST) > 0):
            output = "[INITIAL SETTINGS]\n"
            output += "[NUMBER OF CITIES] " + str(len(CITY_LIST)) + '\n'
            output += "[POPULATION SIZE] " + str(populationSize) + '\n'
            output += "[GENERATIONS] " + str(generations) + '\n'
            output += "---------------------------------------------------------------------------------\n\n"
            TEXT_BOX.insert(END, output)
            start_program(populationSize, generations)
    else:
        showerror("Error", "Invalid settings detected")
        return

CITY_LIST = None

def readCitiesFromFile():
    file = fd.askopenfile()
    if(file):
       cities = []
       cityPositions = []
       for line in file.readlines():
           line = line.strip().split(' ')
           cityPositions.append(line)
       #[city[1:-1].split(', ') for city in re.findall(r'\([^()]*\)', file.read().strip('\n'))]
       for i in range(0, len(cityPositions)):
            cities.append(City(str(i), int(cityPositions[i][0], 10), int(cityPositions[i][1], 10)))
       file.close()
       global CITY_LIST
       CITY_LIST = cities
       TEXT_BOX.insert(END, "[FILE LOADED]\n")

def close():
    ROOT.destroy()
    exit()

def createWindow():
    global ROOT
    ROOT = Tk()
    ROOT.title('zadanie 3')
    mainFrame = Frame(ROOT)
    firstFrame = Frame(mainFrame)
    Button(firstFrame, text="Load file", command=lambda: readCitiesFromFile()).grid(column=0, row=0)
    firstFrame.grid(column=0, row=0)
    secondFrame = Frame(mainFrame)
    randomMode = BooleanVar()
    Checkbutton(firstFrame, text="Random", variable=randomMode).grid(column=1, row=0)

    subFrame = Frame(firstFrame)
    Label(subFrame, text="Number of cities").grid(column=0, row=0)
    maxcities = Entry(subFrame)
    maxcities.grid(column=1, row=0)
    subFrame.grid(column=0, row=1)

    subFrame = Frame(firstFrame)
    Label(subFrame, text="Population size").grid(column=0, row=0)
    amountOfCities = Entry(subFrame)
    amountOfCities.grid(column=1, row=0)
    subFrame.grid(column=0, row=2)

    subFrame = Frame(secondFrame)
    Label(subFrame, text="Generations").grid(column=0, row=0)
    generations = Entry(subFrame)
    generations.grid(column=1, row=0)
    subFrame.grid(column=0, row=3)

    subFrame = Frame(secondFrame)
    Label(subFrame, text="Elitism rate[%]").grid(column=0, row=0)
    elitismRate = IntVar()
    Scale(subFrame,from_=0,to=100, length=400, tickinterval=5, orient='horizontal', variable=elitismRate).grid(column=1, row=0)
    subFrame.grid(column=0, row=4)

    subFrame = Frame(secondFrame)
    Label(subFrame, text="Tournament rate[%]").grid(column=0, row=0)
    tournamentRate = IntVar()
    Scale(subFrame, from_=0, to=100, length=400, tickinterval=5, orient='horizontal', variable=tournamentRate).grid(column=1,row=0)
    subFrame.grid(column=0, row=5)

    subFrame = Frame(secondFrame)
    Label(subFrame, text="Mutation rate[%]").grid(column=0, row=0)
    mutationRate = IntVar()
    Scale(subFrame, from_=0, to=100, length=400, tickinterval=5, orient='horizontal', variable=mutationRate).grid(column=1,row=0)
    subFrame.grid(column=0, row=6)

    subFrame = Frame(secondFrame)
    Label(subFrame, text="One point crossover rate[%]").grid(column=0, row=0)
    onePointCrossRate = IntVar()
    Scale(subFrame, from_=0, to=100, length=400, tickinterval=5, orient='horizontal', variable=onePointCrossRate).grid(column=1, row=0)
    subFrame.grid(column=0, row=7)

    subFrame = Frame(secondFrame)
    Label(subFrame, text="Multipoint crossover rate[%]").grid(column=0, row=0)
    multiCrossRate = IntVar()
    Scale(subFrame, from_=0, to=100, length=400, tickinterval=5, orient='horizontal', variable=multiCrossRate).grid(column=1, row=0)
    subFrame.grid(column=0, row=8)

    secondFrame.grid(column=0, row=2)
    ttk.Separator(mainFrame, orient='horizontal').grid(column=0, row=7, sticky="ew")

    global CANVAS
    frame = Frame(ROOT)
    CANVAS = Canvas(frame, bg="white", bd=2, width=710, height=510)
    CANVAS.pack()
    frame.grid(row=0, column=1)

    global PLOT
    frame = Frame(ROOT)
    global figure
    figure = Figure(figsize=(7, 5), dpi=100)
    PLOT = figure.add_subplot(111)
    graphCanvas = FigureCanvasTkAgg(figure, master=frame)
    graphCanvas.draw()
    graphCanvas.get_tk_widget().pack()
    ani = animation.FuncAnimation(figure, showGraphData, interval=100)
    frame.grid(column=1, row=1)
    Button(mainFrame, text="Start", command=lambda : init(randomMode.get(), maxcities.get(),amountOfCities.get(), generations.get(), [elitismRate.get(), tournamentRate.get(), mutationRate.get(), onePointCrossRate.get(), multiCrossRate.get()])).grid(column=0, row=10)
    Button(mainFrame, text="Stop", command=close).grid(column=0, row=11)
    global TEXT_BOX
    frame = Frame(ROOT)
    TEXT_BOX = Text(frame, width=100, height=25)
    TEXT_BOX.pack()
    frame.grid(column=0, row=1)
    mainFrame.grid(column=0, row=0)


    ROOT.mainloop()

def main():
    createWindow()
    #tsp = TSP("cities.txt")

if __name__ == '__main__':
    main()
