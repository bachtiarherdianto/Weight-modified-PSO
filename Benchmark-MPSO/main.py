"""
MIT License

Copyright (c) 2020 Bachtiar Herdianto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import random
import numpy as np
import matplotlib.pyplot as plt


def Rastrigin (x):          # Rastrigin function
    total = 10*len(x)
    for i in range(len(x)):
        total += x[i]**2 - (10*np.cos(2*np.pi*x[i]))
    return total

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
    def __init__(self, dims, numOfIndiv, numOfEpochs, lower, upper, funct, grad):
        self.swarm_list = [Particle(dims, lower, upper, funct) for i in range(numOfIndiv)]
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

def optimize_orlanj(self, weight, max, min, c1, c2, lr):
    r1 = random.random()
    r2 = random.random()
    funct = self.function
    grad_funct = self.gradien
    boundaries = self.boundaries
    X_epoch = []
    Y_error = []
    for i in range(self.numOfEpochs):
        for j in range(len(self.swarm_list)):
            current_particle = self.swarm_list[j]
            Vcurr = grad_funct(current_particle.position)
            Vcog = r1 * c1 * (current_particle.best_part_pos - current_particle.position)
            Vsos = r2 * c2 * (self.best_swarm_position - current_particle.position)
            Vnew = weight(max, min, i, self.numOfEpochs)*Vcurr + Vcog + Vsos
            new_position = current_particle.position - lr * Vnew
            self.swarm_list[j].setPos(new_position, funct)
            self.swarm_list[j].velocity = Vnew
            self.swarm_list[j].controlPos(boundaries)

            # check the position if it is best for swarm
            if funct(new_position) < self.best_swarm_error or self.best_swarm_error == -1:
                self.best_swarm_position = new_position
                self.best_swarm_error = funct(new_position)

        X_epoch.append(i)
        Y_error.append(self.best_swarm_error)

        if i % 29 == 0:
            print('\nIterasi: {0} \nx1: {1}\nx2: {2}\ny: {3}'.format(i + 1, self.best_swarm_position[0], self.best_swarm_position[1], self.best_swarm_error))

    a, b = np.array(X_epoch), np.array(Y_error)
    plt.plot(a, b)
    plt.xlabel('Iterations')
    plt.ylabel('Objective value (minimization)')
    plt.title('Optimization Graph')
    plt.show()

def Weight(Wmax, Wmin, iteration, maxiter):
    return Wmax - ((Wmax - Wmin)*((1 + iteration)/maxiter))


print('Modified Swarm Intelligent Optimization\nTo optimize Rastrigin function\n')
settingOrlanj = PSO(dims=2, numOfIndiv=20, numOfEpochs=100, lower=-500, upper=500, funct=Rastrigin, grad=grad_Rastrigin)
optimize_orlanj(settingOrlanj, weight=Weight, max=0.8, min=0.6, c1=1.49445, c2=1.49445, lr=0.035)
