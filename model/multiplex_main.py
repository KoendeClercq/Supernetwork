#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 18:14:21 2022

@author: gkdeclercq


----- Model assumptions -----

Multinomial logit is used to calibrate valuations of mode characteristics per cluster. These valuations are used to calculate
the resistance in each link. Shortest path (lowest resistance) for each OD-pair is calculated and updated for each timestep (because of congestion on links).

Density is calculated over the whole length of the link to calculate speed of entering agents.
When density over whole link goes down, speed for all agents go up if this speed is higher
than their speed when entering the link.

BPR-function is used.

Trip generation and distribution is used as input. Modal split and route choice are calculated simultaneously. 
Subsequently, the traffic assignment is fed back to recalculate modal split and route choice in the next timestep.


----- Simulation time -----

Simulation time simple network (1 OD-pair, 5 ('horizontal') edge): appr. 5 min.
Simulation time Sioux Falls (25 OD-pairs, 170 ('horizontal') edges): appr. 1 day on HPC


----- Model decisions -----

                                            OMNITrans       NMM             MATSim          AIMSUN          Supernetwork, Python/NetworkX
Computation time                            Minutes/hours   Minutes/hours   Minutes         Minutes         Minutes/hours
Level of detail                             Macro           Micro           Micro           Micro           Micro/meso?
Congestion                                  Yes             Yes             Yes             Yes             Yes
Spillback                                   Yes             No              Yes             Yes             No
Agents (emergent effects)                   Yes             Yes             Yes             Yes             Yes
Multimodal trips                            No              No              No              No              Yes
Multiple modes                              Yes             Yes             Yes             Yes             Yes
    Shared                                  No              Yes             Yes             Yes             Yes
    Floating                                No              Yes             No              Yes             Yes
Ease of modularity of functionality         No              Yes             No              No              Yes


----- Application of model -----

This model can be used to run 1 scenario, do a scenario-based simulation. It can also be expanded to include multiple 
scenarios, just create an array of the names and configure the attributes of the future mode.


"""

import os
import time as timeOS
from datetime import datetime
import shutil
import csv
import pickle
import ujson as json
import ast
import random
import copy
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import imageio as iio
from pathlib import Path
from itertools import islice
import pdb
import pprint
from functions import *

############################ PARAMETERS #################################

simulate = 1
plotFigures = 0
stats = 1
i = 1
runIterationCntr = 0

networkArray = ['networkDelftCurrentUnimodal', 'networkDelftCurrentMultimodal']
nameFuture = "Electric Step"

for networkName in networkArray:

    print('Started running simulation for', networkName, 'with future mode', nameFuture)

    # Configure trips, OD and simulation parameters
    time = np.arange(400) # 400, Morning peak of 6:00 am till 10:00 am and uniformly distributed starting time of trips between 1 and 200 (first hour of simulation)
    depTime = [1, 400] # 200
    timeStep = 1/100 # [hours] # 1/100 Max. required timestep is 1/100 [hr] based on 2 timesteps on the shortest link (0.375 [km]) with the highest possible speed (35 [km/hr]) for Delft network

    updateResInt = 1 # Interval to update link resistances

    scalingSampleSize = 66 # Aggregate trips + account for aggregated network (estimation), 23922 trips in Sioux Falls, 358823 trips in Delft, Min. scaling factor: 358823/23922 = 15, use same scaling factor as in Sioux Falls: 66.
    scalingSampleSizeTrips = scalingSampleSize # Reduce number of trips (estimation)

    seed = [1234]

    # Parameters for multiplicative MNL with PS-factor
    MNLbeta = [1]
    MNLmu = [1]

    ################# Future mode attributes ########################
        
    cost = [0, 0.05, 0.10]
    costInitial = [0, 4, 8]
    speed = [20] # Assumed, 25 [km/h] is based on max. speed, and with traffic stops, etc., it's assumed that the average speed for each edge is 20 [km/h].
    drivingTask = [1]
    skills = [0]
    weatherProtection = [0]
    luggage = [0]
    shared = [0]
    availability = [1]
    reservation = [0]
    active = [1]
    accessible = [0]
    neutralToFuture = [3/60] # Assumed
    futureToNeutral = [2/60] # Assumed

    futureModeChar = cartesian((cost, speed, drivingTask, skills, weatherProtection, luggage, shared, availability, reservation, active, accessible, neutralToFuture, futureToNeutral, costInitial, seed, MNLbeta, MNLmu))

    startTime = timeOS.time()

    if len(futureModeChar) > 1:
        if simulate:
            links = initializeNetwork(networkName, scalingSampleSizeTrips, futureCharRow[14], depTime)
            simulateNetwork(futureCharRow, links, updateResInt, scalingSampleSize, time, timeStep, networkName, futureCharRow[14], futureCharRow[15], futureCharRow[16], nameFuture)

        if plotFigures:
            plotNetwork(time, 125/10)

        if stats:
            runIterationCntr = statsNetwork(futureCharRow, networkName, nameFuture, scalingSampleSize, scalingSampleSizeTrips, startTime, time, timeStep, runIterationCntr, futureCharRow[15], futureCharRow[16])

        print("Finished running configuration " + str(i) + " out of " + str(len(networkArray)))
        i += 1

    else:
        for futureCharRow in futureModeChar:

            if simulate:
                links = initializeNetwork(networkName, scalingSampleSizeTrips, futureCharRow[14], depTime)
                simulateNetwork(futureCharRow, links, updateResInt, scalingSampleSize, time, timeStep, networkName, futureCharRow[14], futureCharRow[15], futureCharRow[16], nameFuture)

            if plotFigures:
                plotNetwork(time, maxCapacity)

            if stats:
                runIterationCntr = statsNetwork(futureCharRow, networkName, nameFuture, scalingSampleSize, scalingSampleSizeTrips, startTime, time, timeStep, runIterationCntr, futureCharRow[15], futureCharRow[16])

            print("Finished running configuration " + str(i) + " out of " + str(len(networkArray) * len(futureModeChar)))
            i += 1

print("Finished running all network configurations.")
