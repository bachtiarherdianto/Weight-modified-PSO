import numpy as np
import random
import matplotlib.pyplot as plt
import time

def Sphere (x): # Sphere function
    total = 0
    for i in range(len(x)):
        total += x[i]**2
    return total

def Rastrigin (x):  # Rastrigin function
    total = 10*len(x)
    for i in range(len(x)):
        total += x[i]**2 - (10*np.cos(2*np.pi*x[i]))
    return total

def grad_Sphere (x):    # Derivative of Sphere function
    gradient_coordinate = []
    for i in range(len(x)):
        total = 2*x[i]
        gradient_coordinate.append(total)
    return np.array(gradient_coordinate)

def grad_Rastrigin (x):     # Derivative of Rastrigin function
    gradient_coordinate = []
    for i in range(len(x)):
        total = 2*x[i] + 10*2*np.pi*x[i]*np.sin(2*np.pi*x[i])
        gradient_coordinate.append(total)
    return np.array(gradient_coordinate)

class Particle:
    def __init__(self, dim, minx, maxx, error):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=minx, high=maxx, size=dim)
        self.best_part_pos = self.position.copy()
        self.error = error(self.position)
        self.best_part_err = self.error.copy()
    def setPos(self, pos, error):
        self.position = pos
        self.error = error(pos)
        if self.error < self.best_part_err:
            self.best_part_err = self.error
            self.best_part_pos = pos
    def controlPos(self, bounds):
        for i in range(len(bounds)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

class PSO:
    def __init__(self, dims, numOfParticles, numOfEpochs, lower, upper, funct, grad):
        self.swarm_list = [Particle(dims, lower, upper, funct) for i in range(numOfParticles)]
        self.numOfEpochs = numOfEpochs
        self.best_swarm_position = np.random.uniform(low=lower, high=upper, size=dims)
        self.dimension = dims
        self.upper = upper
        self.lower = lower
        self.best_swarm_error = -1
        self.function = funct
        self.gradien = grad
        self.boundaries = []
        for i in range(dims):
            self.boundaries.append((lower, upper))

""" Optimize Function """
def optimize_signature(self, w, c1, c2):
    r1 = random.random()
    r2 = random.random()
    funct = self.function
    X_epoch = []
    Y_error = []
    boundaries = self.boundaries
    for i in range(self.numOfEpochs):
        for j in range(len(self.swarm_list)):
            current_particle = self.swarm_list[j]
            Vcog = r1*c1*(current_particle.best_part_pos - current_particle.position)
            Vsos = r2*c2*(self.best_swarm_position - current_particle.position)
            deltaV = w*current_particle.velocity + Vcog + Vsos   # calculate deltaV
            new_position = current_particle.position + deltaV # calculate the new position
            self.swarm_list[j].setPos(new_position, funct)
            self.swarm_list[j].velocity = deltaV
            self.swarm_list[j].controlPos(boundaries)

            # check the position if it is best for swarm
            if funct(new_position) < self.best_swarm_error or self.best_swarm_error == -1:
                self.best_swarm_position = new_position
                self.best_swarm_error = funct(new_position)
        X_epoch.append(i)
        Y_error.append(self.best_swarm_error)

    a, b = np.array(X_epoch), np.array(Y_error)
    plt.plot(a, b)
    plt.xlabel('Epoch'), plt.ylabel('Objective value (Error)'), plt.title('Report Optimization'), plt.show()
    print('----------------------------\nTotal Epoch: {0} \nBest position: \n[{1}, {2}, {3}, {4}, {5}, {6}] \nBest known error: {7}'.format(
        i + 1, self.best_swarm_position[0], self.best_swarm_position[1], self.best_swarm_position[2],
        self.best_swarm_position[3], self.best_swarm_position[4], self.best_swarm_position[5], self.best_swarm_error))

def optimize_additive(self, weight, max, min, c1, c2):
    r1 = random.random()
    r2 = random.random()
    funct = self.function
    X_epoch = []
    Y_error = []
    boundaries = self.boundaries
    for i in range(self.numOfEpochs):
        for j in range(len(self.swarm_list)):
            current_particle = self.swarm_list[j]
            Vcog = r1*c1*(current_particle.best_part_pos - current_particle.position)
            Vsos = r2*c2*(self.best_swarm_position - current_particle.position)
            deltaV = weight(max, min, i, self.numOfEpochs)*current_particle.velocity + Vcog + Vsos   # calculate deltaV
            new_position = current_particle.position + deltaV # calculate the new position
            self.swarm_list[j].setPos(new_position, funct)
            self.swarm_list[j].velocity = deltaV
            self.swarm_list[j].controlPos(boundaries)

            # check the position if it is best for swarm
            if funct(new_position) < self.best_swarm_error or self.best_swarm_error == -1:
                self.best_swarm_position = new_position
                self.best_swarm_error = funct(new_position)
        X_epoch.append(i)
        Y_error.append(self.best_swarm_error)

    a, b = np.array(X_epoch), np.array(Y_error)
    plt.plot(a, b)
    plt.xlabel('Epoch'), plt.ylabel('Objective value (Error)'), plt.title('Report Optimization'), plt.show()
    print('----------------------------\nTotal Epoch: {0} \nBest position: \n[{1}, {2}, {3}, {4}, {5}, {6}] \nBest known error: {7}'.format(
        i + 1, self.best_swarm_position[0], self.best_swarm_position[1], self.best_swarm_position[2],
        self.best_swarm_position[3], self.best_swarm_position[4], self.best_swarm_position[5], self.best_swarm_error))

def WeightAddictive(Wmax, Wmin, iteration, maxiter):
    return Wmax - (Wmax - Wmin)*((1 + iteration)/maxiter)

""" Running Code 
    General setting:
    Number of dimmensions: 6
    Number of particles: 30
    Number of Max Iterations: 500
    
    Setting no. 1:
    Objective Funct.: Rastrigin
    
    Setting no. 2:
    Objective Funct.: Sphere  """
setting01 = PSO(
    dims=6, 
    numOfParticles=30, 
    numOfEpochs=500, 
    lower=-500, 
    upper=500, 
    funct=Rastrigin, 
    grad=grad_Rastrigin
    )

setting02 = PSO(
    dims=6, 
    numOfParticles=30, 
    numOfEpochs=500, 
    lower=-500, 
    upper=500, 
    funct=Sphere, 
    grad=grad_Sphere
    )

def main():
    print('Original PSO using setting no. 1:')
    start1 = time.time()
    optimize_signature(setting01, w=0.729, c1=1.49445, c2=1.49445)
    end1 = time.time()
    print("processing time:", end1-start1, 'second\n---------------------\n')

    print('Original PSO using setting no. 2:')
    start2 = time.time()
    optimize_signature(setting02, w=1, c1=2, c2=2)
    end2 = time.time()
    print("processing time:", end2-start2, 'second\n---------------------\n')

    print('Modified Weight-Addictive PSO:')
    start = time.time()
    optimize_additive(setting01, weight=WeightAddictive, max=0.9, min=0.1, c1=2, c2=3)
    end = time.time()
    print("processing time:", end - start, 'second\n')


main()