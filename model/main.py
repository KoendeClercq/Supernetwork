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

Spillback is included by calculating speed, densities and then determine the shortest paths. 
So, technically, this is not a full 'spillback' but the route is changed if certain links are too congested.
So, one could argue that this is a way of modeling spillback. REF?

BPR-function is used for cars.

Trip generation and distribution is used as input. Modal split and route choice are calculated simultaneously. 
Subsequently, the traffic assignment is fed back to recalculate modal split and route choice in the next timestep.

Theoretically, agents can get stuck in a loop, therefore a minimum speed has been determined, such that agents will always eventually reach their destination.


----- Simulation time -----

Simulation time simple network:

Car only - simulation time: 2.87 seconds (16 nodes)
Car + metro - simulation time: 6.50 seconds (48 nodes)
Car + metro + future - simulation time: .. seconds

Estimation calculation times network of 1000 nodes:
Car only: (1000/16)^3 * running time simple network = 8 days
One hour of calculation: 172 nodes.

Options to limit simulation time:
- Change timestep (linear)
- Bundle agents (linear)
- Don't update resistance and short path algorithms each timestep (exponential)
- Aggregate network (exponential)
- Parallel computing (linear, but very scalable)
- Parametrize code (e.g., from OD to travel time directly) (exponential)


----- Model decisions -----

                                            OMNITrans       NMM             MATSim          AIMSUN          Supernetwork, Python/NetworkX
Computation time                            Minutes/hours   Minutes/hours   Minutes         Minutes         Minutes/hours
Level of detail                             Macro           Micro           Micro           Micro           Micro/meso?
Congestion                                  Yes             Yes             Yes             Yes             Yes
Spillback                                   Yes             No              Yes             Yes             Yes
Agents (emergent effects)                   Yes             Yes             Yes             Yes             Yes
Multimodal trips                            No              No              No              No              Yes
Multiple modes                              Yes             Yes             Yes             Yes             Yes
    Shared                                  No              Yes             Yes             Yes             Yes
    Floating                                No              Yes             No              Yes             Yes
Ease of modularity of functionality         No              Yes             No              No              Yes


----- Application of model -----

This model can be used to run 1 or multiple scenarios and perform a scenario-based simulation.

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

networkArray = ['DemoFutureMultimodalMultiple4Link1']

nameFuture = "Electric Step"
i = 1
runIterationCntr = 0

for networkName in networkArray:

    print('Started running simulation for', networkName, 'with future mode', nameFuture)

    # Configure trips, OD and simulation parameters
    time = np.arange(100) # Uniformly distributed starting time of trips between 1 and 100
    depTime = [1, 100]
    timeStep = 1/100 # [hours]

    maxCapacity = 125 / 10 # Cars / plot
    updateResInt = 1 # Interval to update link resistances

    scalingSampleSize = 1 # Aggregate trips
    scalingSampleSizeTrips = scalingSampleSize

    seed = [1234]

    # Parameters for multiplicative MNL with PS-factor
    MNLbeta = [1]
    MNLmu = [1]

    ################# Future mode attributes ########################
        
    cost = [0.0, 0.5, 1.0]
    costInitial = [0]
    speed = [10, 20]
    drivingTask = [1]
    skills = [1]
    weatherProtection = [1]
    luggage = [1]
    shared = [0]
    availability = [1]
    reservation = [1]
    active = [1]
    accessible = [0]
    neutralToFuture = [2/60]
    futureToNeutral = [2/60]

    futureModeChar = cartesian((cost, speed, drivingTask, skills, weatherProtection, luggage, shared, availability, reservation, active, accessible, neutralToFuture, futureToNeutral, costInitial, seed, MNLbeta, MNLmu))

    startTime = timeOS.time()

    for futureCharRow in futureModeChar:
        # futureCharRow = futureModeChar[0]

        if simulate:
            links = initializeNetwork(networkName, scalingSampleSizeTrips, futureCharRow[14], depTime)
            simulateNetwork(futureCharRow, links, updateResInt, scalingSampleSize, time, timeStep, networkName, futureCharRow[14], futureCharRow[15], futureCharRow[16], nameFuture)

        if plotFigures:
            plotNetwork(time, maxCapacity)

        if stats:
            runIterationCntr = statsNetwork(futureCharRow, networkName, nameFuture, scalingSampleSize, scalingSampleSizeTrips, startTime, time, timeStep, runIterationCntr, futureCharRow[15], futureCharRow[16])

        print("Finished running configuration " + str(i) + " out of " + str(len(networkArray) * 2))
        i += 1

print("Finished running all network configurations.")
