#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 18:14:21 2022

@author: gkdeclercq
"""

import os
import time as timeOS
from datetime import datetime
import shutil
import multiprocessing as mp
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

############################ FUNCTIONS ##################################

# Get shortest path with pathRes for utility1 and utility2 combined
def shortestPathUtilityCombined(G, recSource, recTarget, cluster):

    allSimplePaths = []
    # Reduce computational complexity by assuming that after each node the shortest path can be found properly based on length only.
    G_temp = copy.deepcopy(G)
    for i in range(100): # Assumed that the shortest path is placed in one of the 100 shortest routes based on distance (see assumption in paper)
        try:
            path = nx.shortest_path(G_temp, str(recSource), str(recTarget), weight='distance', method='bellman-ford')
            G_temp.remove_edge(path[0], path[1])
            allSimplePaths.append(path)
        except:
            pass

    minResistance = -1000000
    # for path in sorted(allSimplePaths): # Combine two resistances to calculate total resistance of route
    for path in allSimplePaths:
        pathLength = 0
        resistance1 = 0 # Summed up for contribution to total resistance
        resistance2 = 0 # Summed up and then divided by the length for contribution to total resistance

        for i in range(len(path)-1):
            resistance1 += G[path[i]][path[i+1]][0]['resistance1' + cluster]
            resistance2 += G[path[i]][path[i+1]][0]['resistance2' + cluster]
            pathLength += G[path[i]][path[i+1]][0]['distance']

        totalResistance = resistance1 + (resistance2 / pathLength)

        if (totalResistance > minResistance or minResistance == -1000000):
            minResistance = totalResistance
            finalPath = path

    return finalPath, minResistance, pathLength


# Determine recursively what the next node is for an OD-pair
def nextNode(G, prevPrevSource, prevSource, recSource, recTarget, cluster, networkName, seed, MNLbeta, MNLmu):

    # Use shortest path & remove link that is used first to create 6 shortest paths
    pathsList = []
    nrOfRoutes = 6
    G_temp = copy.deepcopy(G)
    nrOfEdges = G.number_of_edges()

    # Remove previous link, so that agent doesn't move backwards on route
    if prevSource != "NAN":
        if prevSource != recSource:
            G_temp.remove_node(prevSource)

    if prevPrevSource != "NAN":
        if prevPrevSource != prevSource:
            G_temp.remove_node(prevPrevSource)

    for c in range(nrOfRoutes):
        try:
            path, pathRes, pathLength = shortestPathUtilityCombined(G_temp, recSource, recTarget, cluster)

            # Check if path is multimodal
            if ('Unimodal' in networkName) and (float(path[0]) > 25000) and (float(path[0]) < 100000) and (float(path[1]) < 100000): # Exception for transit implemented, since a switch to another mode is allowed here, if it does not cover the whole network
                indices = [i for i, x in enumerate(path) if float(x) < 25000] # Check if edges are moving 'back' to the OD-layer, representing shifts in mode (which is not allowed when using the unimodal network)
                j = 0
                while (len(indices) > 1) and (j < nrOfEdges): # Repeat until all multimodal options are removed.
                    G_temp.remove_edge(path[indices[1]-1], path[indices[1]])
                    path, pathRes, pathLength = shortestPathUtilityCombined(G_temp, recSource, recTarget, cluster)
                    indices = [i for i, x in enumerate(path) if float(x) < 25000] # Check if edges are moving 'back' to the OD-layer, representing shifts in mode (which is not allowed when using the unimodal network)
                    j += 1

            # Remove first link to create 'nest' in route-set
            G_temp.remove_edge(path[0], path[1])
            pathsList.append([pathRes, path, pathLength])
            
        except:
            pass

    nrOfRoutes = len(pathsList)
    if nrOfRoutes == 0:
        print('Warning: 0 routes found, next node cannot be determined.')
        print(prevPrevSource, prevSource, recSource, recTarget)
            
    # Calculate all PS factors for each of the 6 paths
    PS = [0, 0, 0, 0, 0, 0]
    V = [0, 0, 0, 0, 0, 0]
    for r in range(nrOfRoutes):
        V[r] = pathsList[r][0]

    denominator = 0
    for p in range(nrOfRoutes):
        try:
            tempRes = math.exp(V[p])
        except:
            tempRes = 1000000
        denominator += tempRes

    # Put 6 shortest paths in multiplicative MNL
    P = [1, 1, 1, 1, 1, 1]

    if nrOfRoutes > 1:
        for i in range(nrOfRoutes):
            P[i] = 0

        for r in range(nrOfRoutes):
            try:
                tempRes = math.exp(V[r])
            except:
                tempRes = 1000000

            if r != 0:
                P[r] = (tempRes / denominator) + P[r-1]
            else:
                try:
                    P[r] = tempRes / denominator
                except:
                    # Calculating shortest paths, denominator equal to 0. Last edge in route reached.
                    continue

    # Apply pseudo-randomizer between 0 and 1 to select route
    randValue = random.randrange(0, 100) / 100
    arrayTemp = [i for i, v in enumerate(P) if v > randValue]
    chosenRoute = arrayTemp[0]

    return pathsList[chosenRoute][1][1]

# Resistance (reversed utility) function
def calcResistance(numberOfAgentsOnLink, distance, capacity, ffSpeed, mode, source, target, currentSpeed, t, timeStep, scalingSampleSize, futureCharRow, networkName):

    # Betas per cluster from DCM
    Betas = [[-1.53, -0.156, 0, -0.25, -0.0606, 0.107, 0.193, -0.137, 0.205, 0.186, 0.293, 0.213], # 0
        [-0.0846, -0.4608, 0, 2.11, 2.36, -0.471, -0.755, -1.21, -4.22, -1.34, 0.131, -1.07], # 1
        [-0.12, -0.04566, 0, 0.63, 0.928, 0.0254, -0.143, -0.833, -1.76, 0.0491, -0.195, -0.753], # 2
        [-0.0932, -0.0441, 0, -0.0327, 0.924, 0.486, -1.11, -1.07, 0.395, -1.9, -0.666, -0.248], # 3
        [-0.196, -0.03036, 0, 0.694, -2.69, -0,959, -0.384, 1.75, -1.65, 1.91, 1.39, 0.848], # 4
        [-0.15, -0.03768, 0, -0.136, 1.29, -0.125, 0.458, -1.16, -2.61, 0.314, 0.455, -1.24]] # 5

    # Fundamental diagram, nr of lanes for Delft model based on OmniTRANS Delft model BPR function fitting, see BPR_calibration_all_links.xlsx
    if (distance != 0) and (len(source) > 2):
        
        if 'TestDCM' in networkName:
            # Only used to compare model with DCM
            speed = ffSpeed
        else:
            k_jam = 150 # [veh/km] Assumed maximum density per lane
            density = numberOfAgentsOnLink / distance # [veh/km]
            speed = max(0.1, ffSpeed - (ffSpeed / (k_jam * capacity)) * density) # [km/hr] - Minimum speed of 0.1 km/hr for computational reasons (otherwise agents might stay in (congested) network forever)

            # Fundamental diagram
            critDensity = 25 # [km/hr]
            jamDensity = 150 # [veh/km] Assumed maximum density per lane
            laneCapacity = 2500 # [veh/hr]

            density = numberOfAgentsOnLink / distance # [veh/km]

            if density <= critDensity:
                speed = ffSpeed
            else:
                speed = max((jamDensity - density) * (laneCapacity / (jamDensity - critDensity)) / density, 0.1)

    # speed = max(5, speed)

    # Default values
    cost = 0
    costInitial = 0
    time = 1/3600
    drivingTask = 0
    skills = 0
    weatherProtection = 0
    luggage = 0
    shared = 0
    availability = 1
    reservation = 0
    active = 0
    accessible = 0

    if 'Car' in mode:

        cost = 0.19 # * distance # Assumed 0.19 eur/km
        costInitial = 0
        time = distance / speed
        drivingTask = 1
        skills = 1
        weatherProtection = 1
        luggage = 1
        shared = 0
        availability = 1
        reservation = 1
        active = 0
        accessible = 0

    elif 'Pool' in mode:

        cost = 0.19 / 2 # * distance  # Assumed pick-up and drop, so double the cost - 0.19 / 2 eur/km
        costInitial = 0
        time = distance / speed
        drivingTask = 0
        skills = 0
        weatherProtection = 1
        luggage = 1
        shared = 1
        availability = 0.1
        reservation = 1
        active = 0
        accessible = 1

    elif 'Transit' in mode:

        speed = ffSpeed
        cost = 0.20 # Assumed 0.20 per 1 km
        costInitial = 1
        time = distance / speed
        drivingTask = 0
        skills = 0
        weatherProtection = 1
        luggage = 0.5
        shared = 1
        availability = 0.5
        reservation = 0
        active = 0
        accessible = 1

    elif 'Bicycle' in mode:

        # speed = ffSpeed
        cost = 0
        costInitial = 100 / 7300 # Assumption. 4 trips per day for 5 years, cost of purchase 100 euro
        time = distance / speed
        drivingTask = 1
        skills = 0
        weatherProtection = 0
        luggage = 0
        shared = 0
        availability = 1
        reservation = 1
        active = 1
        accessible = 0

    elif 'Walk' in mode:

        speed = ffSpeed
        cost = 0
        costInitial = 0
        time = distance / speed
        drivingTask = 0
        skills = 0
        weatherProtection = 0
        luggage = 0
        shared = 0
        availability = 1
        reservation = 1
        active = 1
        accessible = 0

    elif 'Future'  in mode:

        speed = futureCharRow[1]
        cost = futureCharRow[0]
        costInitial = futureCharRow[13]
        time = distance / speed
        drivingTask = futureCharRow[2]
        skills = futureCharRow[3]
        weatherProtection = futureCharRow[4]
        luggage = futureCharRow[5]
        shared = futureCharRow[6]
        availability = futureCharRow[7]
        reservation = futureCharRow[8]
        active = futureCharRow[9]
        accessible = futureCharRow[10]


    if mode == 'CarToNeutral':
        time = 2 / 60 * 3 # [hour]

    elif mode == 'CarpoolToNeutral':
        time = 2 / 60 * 3 # [hour]

    elif mode == 'TransitToNeutral':
        time = 0.1 / 60 * 3 # [hour]

    elif mode == 'BicycleToNeutral':
        time = 1 / 60 * 3 # [hour]

    elif mode == 'WalkToNeutral':
        time = 0.1 / 60 * 3 # [hour]

    elif mode == 'FutureToNeutral':
        time = futureCharRow[12] * 3 # [hour]

    elif mode == 'TransitToTram001':
        time = 5 / 60 * 3 # [hour] Freq. based - every 10 min

    elif mode == 'TransitToTram019':
        time = 10 / 60 * 3 # [hour] Freq. based - every 20 min

    elif mode == 'TransitToBus455':
        time = 7.5 / 60 * 3 # [hour] Freq. based - every 15 min

    elif mode == 'TransitToBus040':
        time = 15 / 60 * 3 # [hour] Freq. based - every 30 min

    elif mode == 'TransitToBus060':
        time = 7.5 / 60 * 3 # [hour] Freq. based - every 15 min

    elif mode == 'TransitToBus061':
        time = 15 / 60 * 3 # [hour] Freq. based - every 30 min

    elif mode == 'TransitToBus062':
        time = 15 / 60 * 3 # [hour] Freq. based - every 30 min

    elif mode == 'TransitToBus063':
        time = 15 / 60 * 3 # [hour] Freq. based - every 30 min

    elif mode == 'TransitToBus064':
        time = 7.5 / 60 * 3 # [hour] Freq. based - every 15 min

    elif mode == 'TransitToBus069':
        time = 7.5 / 60 * 3 # [hour] Freq. based - every 15 min

    elif mode == 'TransitToBus174':
        time = 15 / 60 * 3 # [hour] Freq. based - every 30 min

    elif mode == 'TransitToBus033':
        time = 15 / 60 * 3 # [hour] Freq. based - every 30 min

    elif mode == 'TransitToBus037':
        time = 15 / 60 * 3 # [hour] Freq. based - every 30 min

    elif mode == 'TransitToBus053':
        if t < 300:
            time = 15 / 60 * 3 # [hour] Freq. based - every 30 min
        else:
            time = 999999999 # Not available after 9:00am

    elif mode == 'TransitToTrain015':
        time = 7.5 / 60 * 3 # [hour] Freq. based - every 30 min  

    elif mode == 'NeutralToCar':
        time = 2 / 60 * 3 # [hour]

    elif mode == 'NeutralToCarpool':
        time = 10 / 60 * 3 # [hour]

    elif mode == 'NeutralToTransit':
        time = 0.1 / 60 * 3 # [hour]

    elif mode == 'NeutralToBicycle':
        time = 1 / 60 * 3 # [hour]

    elif mode == 'NeutralToWalk':
        time = 0.1 / 60 * 3 # [hour]

    elif mode == 'NeutralToFuture':
        time = futureCharRow[11] * 3 # [hour]

    else: # For all transit to transit base layer 'XXXToTransit'
        time = 0.1 / 60 * 3 # [hour]


    # Calculate resistance per cluster
    resistance1 = [0, 0, 0, 0, 0, 0]
    resistance2 = [0, 0, 0, 0, 0, 0]
    
    for i in range(6):
        B = Betas[i]

        if 'NeutralTo' in mode:
            if availability != 0:
                if 'TestDCM' in networkName:
                    utility1 = B[0] * costInitial + B[4] * skills + B[8] * availability + B[9] * reservation
                else:
                    utility1 = B[0] * costInitial + B[1] * 60 * time + B[4] * skills + B[8] * availability + B[9] * reservation
            else:
                utility1 = -1000000 # Means that availability == 0, is a high resistance, such that this route cannot be chosen
            utility2 = 0

            if time != 0:
                speed = 3 * distance / time # * 3 to compensate for 3x in utility calculation when shifting modes ('transfer time') to get the right speed to simulate the time on the 'transfer'-link
            else:
                speed = 3 * distance / (1/3600) # 1 sec, For computational reasons

        elif 'ToNeutral' in mode:
            if availability != 0:
                if 'TestDCM' in networkName:
                    utility1 = 0
                else:
                    utility1 = B[1] * 60 * time
            else:
                utility1 = -1000000 # Means that availability == 0, is a high resistance, such that this route cannot be chosen
            
            utility2 = 0

            if time != 0:
                speed = 3 * distance / time # * 3 to compensate for 3x in utility calculation when shifting modes ('transfer time') to get the right speed to simulate the time on the 'transfer'-link
            else:
                speed = 3 * distance / (1/3600) # 1 sec, For computational reasons

        elif 'OD' in mode:
            utility1 = 0.0001
            utility2 = 0.0001
            time = 1 / 70
            speed = distance / time

        else:
            utility1 = distance * (B[0] * cost + B[1] * 60 / speed) # Can be summed up in final resistance calculation
            utility2 = distance * (B[3] * drivingTask + B[5] * weatherProtection + B[6] * luggage + B[7] * shared + B[10] * active + B[11] * accessible) # Can be summed up in final resistance calculation then divide by total length of route

        try:
            utility1 = utility1.item()
        except:
            pass 

        try:
            utility2 = utility2.item()
        except:
            pass

        resistance1[i] = utility1
        resistance2[i] = utility2

    try:
        speed = speed.item()
    except:
        pass 

    try:
        numberOfAgentsOnLink = numberOfAgentsOnLink.item()
    except:
        pass 

    return resistance1, resistance2, speed, numberOfAgentsOnLink


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    Source: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def initializeNetwork(networkName, scalingSampleSizeTrips, seed, depTime):

    # Generate network
    # Input networkName to select right network & scaling factor to reduce sample size of trips (and speed up simulation).

    # Initialize trips
    f = open('dataset/tripsPre.json')
    data = json.load(f)
    trips = []

    # random.seed(seed)
    data = random.sample(data, int(len(data) / scalingSampleSizeTrips)) 

    counter = 0
    for i in data:
        dictTrips = {
            "id": counter,
            "origin": i['origin'],
            "destination": i['destination'],
            "type": i['cluster'],
            "depTime": random.randrange(depTime[0], depTime[1]) # In timesteps from start simulation # 
        }
        trips.append(dictTrips)
        counter += 1

    with open('data/trips.json', 'w', encoding='utf-8') as f:
        json.dump(trips, f, ensure_ascii=False, indent=4)


    # Create CSV

    if networkName != 'networkDelft':

        # Name, type (node/edge), positions for nodes,
        # start and end point, resistance (inversed utility), capacity (PCU), distance (km), average speed and default speed (without congestion) (km/h) for links.
        header = ['name', 'type', 'pos_x', 'pos_y', 'source', 'target', 'resistance', 'capacity', 'distance', 'speed', 'defSpeed', 'mode']
        numberOfNodes = 16

        with open('data/simpleNetwork.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            if networkName == 'carSimple':

                # Create nodes
                for i in range(3):
                    data = [i, 'node', 0, i, 0, 0]
                    writer.writerow(data)

                # Create links car
                for i in range(2):
                    data = [1001+i, 'link', 0, 0, 0+i, 1+i, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                    writer.writerow(data)

            if networkName == 'metroSimple':

                # Create nodes
                for i in range(3):
                    data = [i, 'node', 0, i, 0, 0]
                    writer.writerow(data)

                # Create links metro
                for i in range(2):
                    data = [2001+i, 'link', 0, 0, 0+i, 1+i, 6, 2500, 10, 30, 30, 'metro']
                    writer.writerow(data)

            if networkName == 'carOnly':

                # Create nodes
                for i in range(numberOfNodes):
                    data = [i, 'node', i%4/3, i//4, 0, 0]
                    writer.writerow(data)

                # Create links car
                for i in range(numberOfNodes - 1):
                    if (i != 3) and (i != 7) and (i != 11):
                        # One-way
                        data = [i, 'link', 0, 0, i, i+1, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)
                        # Other-way
                        data = [i, 'link', 0, 0, i+1, i, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)

                    if (i < numberOfNodes - 4):
                        # One-way
                        data = [i, 'link', 0, 0, i, i+4, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)
                        # Other-way
                        data = [i, 'link', 0, 0, i+4, i, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)

            # Create network with 2 modes & 'middle-layer' with just OD-nodes

            if networkName == 'carMetro':

                # Create nodes
                for i in range(numberOfNodes):
                    data = [i, 'node', i%4/3, i//4, 0, 0]
                    writer.writerow(data)

                for i in range(1000, 1000+numberOfNodes):
                    data = [i, 'node', i%4/3, (i-1000)//4, 0, 0]
                    writer.writerow(data)

                for i in range(2000, 2000+numberOfNodes):
                    data = [i, 'node', i%4/3, (i-2000)//4, 0, 0]
                    writer.writerow(data)


                # Create links car
                layerNumber = 1000
                for i in range(layerNumber, layerNumber+numberOfNodes - 1):
                    if (i != layerNumber+3) and (i != layerNumber+7) and (i != layerNumber+11):
                        # One-way
                        data = [i, 'link', 0, 0, i, i+1, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)
                        # Other-way
                        data = [i, 'link', 0, 0, i+1, i, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)

                    if (i < layerNumber+numberOfNodes - 4):
                        # One-way
                        data = [i, 'link', 0, 0, i, i+4, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)
                        # Other-way
                        data = [i, 'link', 0, 0, i+4, i, 6, 2000, 5, 30, 30, 'car'] # Based on Snelder et al. (2019) Mobility impacts of automated driving and shared mobility
                        writer.writerow(data)

                for i in range(numberOfNodes):
                    # One-way
                    data = [i, 'link', 0, 0, i, i+layerNumber, 1, 0, 0.1, 20, 20, 'neutralToCar']
                    writer.writerow(data)
                    # Other-way
                    data = [i, 'link', 0, 0, i+layerNumber, i, 1, 0, 0.1, 20, 20, 'carToNeutral']
                    writer.writerow(data)


                # Create links metro
                layerNumber = 2000
                for i in range(layerNumber, layerNumber+numberOfNodes - 1):
                    if (i != layerNumber+3) and (i != layerNumber+7) and (i != layerNumber+11):
                        # One-way
                        data = [i, 'link', 0, 0, i, i+1, 6, 2500, 10, 60, 60, 'metro']
                        writer.writerow(data)
                        # Other-way
                        data = [i, 'link', 0, 0, i+1, i, 6, 2500, 10, 60, 60, 'metro']
                        writer.writerow(data)

                    if (i < layerNumber+numberOfNodes - 4):
                        # One-way
                        data = [i, 'link', 0, 0, i, i+4, 6, 2500, 10, 60, 60, 'metro']
                        writer.writerow(data)
                        # Other-way
                        data = [i, 'link', 0, 0, i+4, i, 6, 2500, 10, 60, 60, 'metro']
                        writer.writerow(data)

                for i in range(numberOfNodes):
                    # One-way
                    data = [i, 'link', 0, 0, i, i+layerNumber, 1, 0, 0.1, 20, 20, 'neutralToMetro']
                    writer.writerow(data)
                    # Other-way
                    data = [i, 'link', 0, 0, i+layerNumber, i, 1, 0, 0.1, 20, 20, 'metroToNeutral']
                    writer.writerow(data)

            if networkName == 'simpleSix':

                # Modes included: car, carpool, transit, bicycle, walk, future

                #          2      9     16           <-- neutral
                #     __-- 3 --- 10 --- 17 --__      <-- car
                #   0 ---- 4 --- 11 --- 18 ---- 1    <-- carpool
                #   |\\--- 5 --- 12 --- 19 ---//|    <-- transit
                #    \\--- 6 --- 13 --- 20 ---//     <-- bicycle
                #     ---- 7 --- 14 --- 21 ----      <-- walk
                #      --- 8 --- 15 --- 22 ---       <-- future mode

                # Create nodes
                for i in range(2):
                    data = [i, 'node', i*4, 0, 0, 0]
                    writer.writerow(data)

                for y in range(7):
                    for x in range(3):
                        data = [2+x*7+y, 'node', x+1, 3-y, 0, 0]
                        writer.writerow(data)

                # Create links
                layerNumber = [1000, 2000, 3000, 4000, 5000, 6000]
                locations = [[0, 3, 10, 17, 1], [0, 4, 11, 18, 1], [0, 5, 12, 19, 1], [0, 6, 13, 20, 1], [0, 7, 14, 21, 1], [0, 8, 15, 22, 1]]

                resistance = [6, 6, 6, 6, 6, 6]
                resistanceToNeutral = [1, 1, 1, 1, 1, 1]
                capacity = [2000, 500, 2500, 5000, 10000, 3000]
                distance = [5, 5, 5, 5, 5, 5]
                speed = [30, 30, 50, 25, 15, 40]
                defSpeed = [30, 30, 50, 25, 15, 40]
                mode = ['Car', 'Pool', 'Transit', 'Bicycle', 'Walk', 'Future'] # Pool = carpool, coding reasons, just called pool.

                for j in range(6):
                    for i in range(4):
                        data = [layerNumber[j]+i, 'link', 0, 0, locations[j][0+i], locations[j][1+i], resistance[j], capacity[j], distance[j], speed[j], defSpeed[j], mode[j]]
                        writer.writerow(data)

                    for i in range(3):
                        # One-way
                        data = [layerNumber[j]+4+i, 'link', 0, 0, locations[j][1]+7*i, 2+7*i, resistanceToNeutral[j], capacity[j], distance[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                        writer.writerow(data)
                        # other-way
                        data = [layerNumber[j]+7+i, 'link', 0, 0, 2+7*i, locations[j][1]+7*i, resistanceToNeutral[j], capacity[j], distance[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                        writer.writerow(data)

            if networkName == 'testNetwork':

                # Modes is car only

                # Create nodes
                data = [0, 'node', -2, -2, 0, 0]
                writer.writerow(data)

                for i in range(15):
                    data = [i+1, 'node', 0, i, 0, 0]
                    writer.writerow(data)

                for i in range(5):
                    data = [i+16, 'node', i+1, i+15, 0, 0]
                    writer.writerow(data)

                    data = [i+23, 'node', -i-1, i+15, 0, 0]
                    writer.writerow(data)

                    data = [i+31, 'node', -10+i, i+14, 0, 0]
                    writer.writerow(data)

                data = [21, 'node', 8, 22, 0, 0]
                writer.writerow(data)

                data = [22, 'node', 10, 23, 0, 0]
                writer.writerow(data)

                data = [28, 'node', -8, 22, 0, 0]
                writer.writerow(data)

                data = [29, 'node', -9, 24, 0, 0]
                writer.writerow(data)

                data = [30, 'node', -12, 15, 0, 0]
                writer.writerow(data)


                # Create links
                resistance = [6]
                capacity = [2000]
                distance = [2]
                speed = [60]
                defSpeed = [60]
                mode = ['Car']
                j = 0

                for i in range(22):
                    data = [1000+i, 'link', 0, 0, i, i+1, resistance[j], capacity[j], distance[j], speed[j], defSpeed[j], mode[j]]
                    writer.writerow(data)

                data = [1000+i, 'link', 0, 0, 15, 24, resistance[j], capacity[j], distance[j], speed[j], defSpeed[j], mode[j]]
                writer.writerow(data)

                for i in range(5):
                    data = [1000+i, 'link', 0, 0, i+24, i+25, resistance[j], capacity[j], distance[j], speed[j], defSpeed[j], mode[j]]
                    writer.writerow(data)

                for i in range(5):
                    data = [1000+i, 'link', 0, 0, i+30, i+31, resistance[j], capacity[j], distance[j], speed[j], defSpeed[j], mode[j]]
                    writer.writerow(data)

                data = [1000+i, 'link', 0, 0, 35, 27, resistance[j], capacity[j], distance[j], speed[j], defSpeed[j], mode[j]]
                writer.writerow(data)


######################################################################################################################################################################################################


            if 'Demo' in networkName:

                # Modes included: car, carpool, transit, bicycle, walk, future

                # OD/centroids layer:   00000
                # Neutral layer:        20000
                # Car layer:            30000
                # Carpool layer:        40000
                # Transit layer:        50000
                # Bicycle layer:        60000
                # Walk layer:           70000
                # Future layer:         80000

                if 'Future' in networkName:
                    nrOfModes = 6
                elif 'Current' in networkName:
                    nrOfModes = 5

                if 'Multimodal' in networkName:
                    neutralLayerFactor = 20000
                else:
                    neutralLayerFactor = 0

                data = [1001, 'node', 0, 0, 0, 0]
                writer.writerow(data)
                data = [1002, 'node', 10, 0, 0, 0]
                writer.writerow(data)

                if 'Multiple' in networkName:

                    data = [1003, 'node', 5, -5, 0, 0]
                    writer.writerow(data)
                    data = [1004, 'node', 5, 5, 0, 0]
                    writer.writerow(data)

                if 'Multimodal' in networkName:

                    # Create nodes neutral layer
                    data = [21001, 'node', 0, 0, 0, 0]
                    writer.writerow(data)
                    data = [21002, 'node', 10, 0, 0, 0]
                    writer.writerow(data)

                    if 'Multiple' in networkName:
                        
                        data = [21003, 'node', 5, -5, 0, 0]
                        writer.writerow(data)
                        data = [21004, 'node', 5, 5, 0, 0]
                        writer.writerow(data)



                
                if 'Single' in networkName:

                    # Create centroids
                    zones = [1001, 1002]
                    zonesOD = [1001, 1002]

                    zonesFrom = [1001]
                    zonesTo   = [1002]

                if 'Multiple2' in networkName:

                    # Create centroids
                    zones = [1001, 1002, 1003, 1004]
                    zonesOD = [1001, 1002, 1003, 1004]

                    zonesFrom = [1001, 1001, 1003, 1004]
                    zonesTo   = [1003, 1004, 1002, 1002]

                if 'Multiple4' in networkName:

                    # Create centroids
                    zones = [1001, 1002, 1003, 1004]
                    zonesOD = [1001, 1002, 1003, 1004]

                    zonesFrom = [1001, 1001, 1003, 1004, 1003, 1004]
                    zonesTo   = [1003, 1004, 1002, 1002, 1004, 1003]

                if 'Link1' in networkName:

                    distance  = [1, 1, 1, 1, 1, 1]

                if 'Link2' in networkName:

                    distance  = [2, 2, 2, 2, 2, 2]

                if 'Link3' in networkName:

                    distance  = [3, 3, 3, 3, 3, 3]

                if 'Link4' in networkName:

                    distance  = [4, 4, 4, 4, 4, 4]

                if 'Link5' in networkName:

                    distance  = [5, 5, 5, 5, 5, 5]

                if 'Link6' in networkName:

                    distance  = [6, 6, 6, 6, 6, 6]

                layerNumber = [30000, 40000, 50000, 60000, 70000, 80000]
                locations = [zones, zones, zones, zones, zones, zones]

                resistance = [6, 6, 6, 6, 6, 6]
                resistanceToNeutral = [1, 1, 1, 1, 1, 1]
                neutralToResistance = [1, 1, 1, 1, 1, 1]
                distanceNeutralToMode = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
                capacity = [2000, 2000, 2000, 2000, 2000, 2000]
                speed = [34.668441679929636, 30.67680672911914, 29.074451916316153, 14.621403427954009, 5, 10] # From ODIN 2019
                defSpeed = [34.668441679929636, 30.67680672911914, 29.074451916316153, 14.621403427954009, 5, 10] # From ODIN 2019
                mode = ['Car', 'Pool', 'Transit', 'Bicycle', 'Walk', 'Future'] # Pool = carpool, coding reasons, just called pool.

                for j in range(nrOfModes):

                    # Create nodes
                    data = [layerNumber[j]+1001, 'node', 0, 0, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1002, 'node', 10, 0, 0, 0]
                    writer.writerow(data)

                    if 'Multiple' in networkName:

                        data = [layerNumber[j]+1003, 'node', 5, -5, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004, 'node', 5, 5, 0, 0]
                        writer.writerow(data)

                    # Create links
                    for k in range(len(zonesFrom)):
                        # One-way
                        data = [layerNumber[j]+k, 'link', 0, 0, layerNumber[j]+zonesFrom[k], layerNumber[j]+zonesTo[k], resistance[j], capacity[j], distance[k], speed[j], defSpeed[j], mode[j]]
                        writer.writerow(data)
                        # # Other-way (backwards, demo networks have only 1 direction: forward)
                        # data = [layerNumber[j]+k+5000, 'link', 0, 0, layerNumber[j]+zonesTo[k], layerNumber[j]+zonesFrom[k], resistance[j], capacity[j], distance[k], speed[j], defSpeed[j], mode[j]]
                        # writer.writerow(data)

                    if 'Multimodal' in networkName:
                        for i in zones:
                            # One-way
                            data = [layerNumber[j]+i, 'link', 0, 0, layerNumber[j]+i, neutralLayerFactor+i, resistance[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000, 'link', 0, 0, neutralLayerFactor+i, layerNumber[j]+i, resistance[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                            writer.writerow(data)
                    else:
                        for i in zonesOD:
                            # One-way
                            data = [layerNumber[j]+i, 'link', 0, 0, layerNumber[j]+i, 0+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000, 'link', 0, 0, 0+i, layerNumber[j]+i, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                            writer.writerow(data)

                if 'Multimodal' in networkName:
                    # OD centroids to neutral layer
                    for i in zonesOD:
                        # One-way
                        data = [10000+i, 'link', 0, 0, i, 20000+i, 6, 2000, 0.001, 5, 5, 'ODToNeutral']
                        writer.writerow(data)
                        # Other-way
                        data = [10000+i+5000, 'link', 0, 0, 20000+i, i, 6, 2000, 0.001, 5, 5, 'NeutralToOD']
                        writer.writerow(data)


######################################################################################################################################################################################################


            if 'SiouxFalls' in networkName:

                # Modes included: car, carpool, transit, bicycle, walk, future

                # OD/centroids layer:   00000
                # Neutral layer:        20000
                # Car layer:            30000
                # Carpool layer:        40000
                # Transit layer:        50000
                # Bicycle layer:        60000
                # Walk layer:           70000
                # Future layer:         80000

                if 'Future' in networkName:
                    nrOfModes = 6
                elif 'Current' in networkName:
                    nrOfModes = 5

                if 'Multimodal' in networkName:
                    neutralLayerFactor = 20000
                else:
                    neutralLayerFactor = 0

                data = [1001, 'node', 50000, 510000, 0, 0]
                writer.writerow(data)
                data = [1002, 'node', 320000, 510000, 0, 0]
                writer.writerow(data)
                data = [1003, 'node', 50000, 440000, 0, 0]
                writer.writerow(data)
                data = [1004, 'node', 130000, 440000, 0, 0]
                writer.writerow(data)
                data = [1005, 'node', 220000, 440000, 0, 0]
                writer.writerow(data)
                data = [1006, 'node', 320000, 440000, 0, 0]
                writer.writerow(data)
                data = [1007, 'node', 420000, 380000, 0, 0]
                writer.writerow(data)
                data = [1008, 'node', 320000, 380000, 0, 0]
                writer.writerow(data)
                data = [1009, 'node', 220000, 380000, 0, 0]
                writer.writerow(data)
                data = [1010, 'node', 220000, 320000, 0, 0]
                writer.writerow(data)
                data = [1011, 'node', 130000, 320000, 0, 0]
                writer.writerow(data)
                data = [1012, 'node', 50000, 320000, 0, 0]
                writer.writerow(data)
                data = [1013, 'node', 50000, 50000, 0, 0]
                writer.writerow(data)
                data = [1014, 'node', 130000, 190000, 0, 0]
                writer.writerow(data)
                data = [1015, 'node', 220000, 190000, 0, 0]
                writer.writerow(data)
                data = [1016, 'node', 320000, 320000, 0, 0]
                writer.writerow(data)
                data = [1017, 'node', 320000, 260000, 0, 0]
                writer.writerow(data)
                data = [1018, 'node', 420000, 320000, 0, 0]
                writer.writerow(data)
                data = [1019, 'node', 320000, 190000, 0, 0]
                writer.writerow(data)
                data = [1020, 'node', 320000, 50000, 0, 0]
                writer.writerow(data)
                data = [1021, 'node', 220000, 50000, 0, 0]
                writer.writerow(data)
                data = [1022, 'node', 220000, 130000, 0, 0]
                writer.writerow(data)
                data = [1023, 'node', 130000, 130000, 0, 0]
                writer.writerow(data)
                data = [1024, 'node', 130000, 50000, 0, 0]
                writer.writerow(data)


                if 'Multimodal' in networkName:

                    # Create nodes neutral layer
                    data = [21001, 'node', 50000, 510000, 0, 0]
                    writer.writerow(data)
                    data = [21002, 'node', 320000, 510000, 0, 0]
                    writer.writerow(data)
                    data = [21003, 'node', 50000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [21004, 'node', 130000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [21005, 'node', 220000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [21006, 'node', 320000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [21007, 'node', 420000, 380000, 0, 0]
                    writer.writerow(data)
                    data = [21008, 'node', 320000, 380000, 0, 0]
                    writer.writerow(data)
                    data = [21009, 'node', 220000, 380000, 0, 0]
                    writer.writerow(data)
                    data = [21010, 'node', 220000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [21011, 'node', 130000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [21012, 'node', 50000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [21013, 'node', 50000, 50000, 0, 0]
                    writer.writerow(data)
                    data = [21014, 'node', 130000, 190000, 0, 0]
                    writer.writerow(data)
                    data = [21015, 'node', 220000, 190000, 0, 0]
                    writer.writerow(data)
                    data = [21016, 'node', 320000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [21017, 'node', 320000, 260000, 0, 0]
                    writer.writerow(data)
                    data = [21018, 'node', 420000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [21019, 'node', 320000, 190000, 0, 0]
                    writer.writerow(data)
                    data = [21020, 'node', 320000, 50000, 0, 0]
                    writer.writerow(data)
                    data = [21021, 'node', 220000, 50000, 0, 0]
                    writer.writerow(data)
                    data = [21022, 'node', 220000, 130000, 0, 0]
                    writer.writerow(data)
                    data = [21023, 'node', 130000, 130000, 0, 0]
                    writer.writerow(data)
                    data = [21024, 'node', 130000, 50000, 0, 0]
                    writer.writerow(data)

                # Create centroids
                zones = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024]
                zonesOD = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024]

                zonesFromTemp = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 24, 24, 24]
                zonesFrom = [x+1000 for x in zonesFromTemp]

                zonesToTemp   = [2, 3, 1, 6, 1, 4, 12, 3, 5, 11, 4, 6, 9, 2, 5, 8, 8, 18, 6, 7, 9, 16, 5, 8, 10, 9, 11, 15, 16, 17, 4, 10, 12, 14, 3, 11, 13, 12, 24, 11, 15, 23, 10, 14, 19, 22, 8, 10, 17, 18, 10, 16, 19, 7, 16, 20, 15, 17, 20, 18, 19, 21, 22, 20, 22, 24, 15, 20, 21, 23, 14, 22, 24, 13, 21, 23]
                zonesTo = [x+1000 for x in zonesToTemp]

                layerNumber = [30000, 40000, 50000, 60000, 70000, 80000]
                locations = [zones, zones, zones, zones, zones, zones]

                distance  = [3, 2, 3, 2.5, 2, 2, 2, 2, 1, 3, 1, 2, 2.5, 2.5, 2, 1, 1.5, 1, 1, 1.5, 5, 2.5, 2.5, 5, 1.5, 1.5, 2.5, 3, 2, 4, 3, 2.5, 3, 2, 2, 3, 1.5, 1.5, 2, 2, 2.5, 2, 3, 2.5, 1.5, 1.5, 2.5, 2, 1, 1.5, 4, 1, 1, 1, 1.5, 2, 1.5, 1, 2, 2, 2, 3, 2.5, 3, 1, 1.5, 1.5, 2.5, 1, 2, 2, 2, 1, 2, 1.5, 1]
                capacity = [25900, 23403, 25900, 4958, 23403, 17111, 23403, 17111, 17783, 4909, 17783, 4948, 10000, 4958, 4948, 4899, 7842, 23403, 4899, 7842, 5050, 5046, 10000, 5050, 13916, 13916, 10000, 13512, 4855, 4994, 4909, 10000, 4909, 4877, 23403, 4909, 25900, 25900, 5091, 4877, 5128, 4925, 13512, 5128, 14565, 9599, 5046, 4855, 5230, 19680, 4994, 5230, 4824, 23403, 19680, 23403, 14565, 4824, 5003, 23403, 5003, 5060, 5076, 5060, 5230, 4885, 9599, 5076, 5230, 5000, 4925, 5000, 5079, 5091, 4885, 5079]
                # Nr of lanes for Fundamental Diagram
                lanes    = [1,   3,   3,   2,   2,   2,   2,   1,   1,   2,   2,   2,   2,   3,   3,   1,   1,   3,   3,   2,   2,   2,   2,   2,   1,   1,   2,   2,   2,   3,   3,   1,   1,   3,   2,   1,   2,   3,   3,   2,   2,   2,   1,   2,   2,   2,   2,   2,   3,   1,   1,   3,   3,   2,   1,   2,   3,   1,   2,   2,   2,   2,   2,   2,   2,   2,   3,   1,   1,   3,   3,   2,   2,   2,   2,   1]

                resistance = [6, 6, 6, 6, 6, 6]
                resistanceToNeutral = [1, 1, 1, 1, 1, 1]
                neutralToResistance = [1, 1, 1, 1, 1, 1]
                distanceNeutralToMode = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
                speed = [34.668441679929636, 30.67680672911914, 29.074451916316153, 14.621403427954009, 5, 10] # From ODIN 2019
                defSpeed = [34.668441679929636, 30.67680672911914, 29.074451916316153, 14.621403427954009, 5, 10] # From ODIN 2019
                mode = ['Car', 'Pool', 'Transit', 'Bicycle', 'Walk', 'Future'] # Pool = carpool, coding reasons, just called pool.

                for j in range(nrOfModes):

                    # Create nodes
                    data = [layerNumber[j]+1001, 'node', 50000, 510000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1002, 'node', 320000, 510000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1003, 'node', 50000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1004, 'node', 130000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1005, 'node', 220000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1006, 'node', 320000, 440000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1007, 'node', 420000, 380000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1008, 'node', 320000, 380000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1009, 'node', 220000, 380000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1010, 'node', 220000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1011, 'node', 130000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1012, 'node', 50000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1013, 'node', 50000, 50000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1014, 'node', 130000, 190000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1015, 'node', 220000, 190000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1016, 'node', 320000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1017, 'node', 320000, 260000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1018, 'node', 420000, 320000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1019, 'node', 320000, 190000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1020, 'node', 320000, 50000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1021, 'node', 220000, 50000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1022, 'node', 220000, 130000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1023, 'node', 130000, 130000, 0, 0]
                    writer.writerow(data)
                    data = [layerNumber[j]+1024, 'node', 130000, 50000, 0, 0]
                    writer.writerow(data)

                    # Create links
                    for k in range(len(zonesFrom)):
                        # One-way
                        data = [layerNumber[j]+k, 'link', 0, 0, layerNumber[j]+zonesFrom[k], layerNumber[j]+zonesTo[k], resistance[j], lanes[k], distance[k], speed[j], defSpeed[j], mode[j]]
                        writer.writerow(data)
                        # Other-way (backwards, demo networks have only 1 direction: forward)
                        # data = [layerNumber[j]+k+5000, 'link', 0, 0, layerNumber[j]+zonesTo[k], layerNumber[j]+zonesFrom[k], resistance[j], lanes[k], distance[k], speed[j], defSpeed[j], mode[j]]
                        # writer.writerow(data)

                    if 'Multimodal' in networkName:
                        for i in zones:
                            # One-way
                            data = [layerNumber[j]+i, 'link', 0, 0, layerNumber[j]+i, neutralLayerFactor+i, resistance[j], lanes[k], distanceNeutralToMode[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000, 'link', 0, 0, neutralLayerFactor+i, layerNumber[j]+i, resistance[j], lanes[k], distanceNeutralToMode[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                            writer.writerow(data)
                    else:
                        for i in zonesOD:
                            # One-way
                            data = [layerNumber[j]+i, 'link', 0, 0, layerNumber[j]+i, 0+i, resistanceToNeutral[j], lanes[k], distanceNeutralToMode[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000, 'link', 0, 0, 0+i, layerNumber[j]+i, neutralToResistance[j], lanes[k], distanceNeutralToMode[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                            writer.writerow(data)

                if 'Multimodal' in networkName:
                    # OD centroids to neutral layer
                    for i in zonesOD:
                        # One-way
                        data = [10000+i, 'link', 0, 0, i, 20000+i, 6, 2000, 0.001, 5, 5, 'ODToNeutral']
                        writer.writerow(data)
                        # Other-way
                        data = [10000+i+5000, 'link', 0, 0, 20000+i, i, 6, 2000, 0.001, 5, 5, 'NeutralToOD']
                        writer.writerow(data)




######################################################################################################################################################################################################


            if 'Delft' in networkName:

                # Modes included: car, carpool, transit, bicycle, walk, future

                # OD/centroids layer:   00000
                # Neutral layer:        20000
                # Car layer:            30000
                # Carpool layer:        40000
                # Transit layer:        50000
                # Bicycle layer:        60000
                # Walk layer:           70000
                # Future layer:         80000

                if 'Future' in networkName:
                    nrOfModes = 6
                elif 'Current' in networkName:
                    nrOfModes = 5

                if 'Multimodal' in networkName:
                    neutralLayerFactor = 20000
                else:
                    neutralLayerFactor = 0

                
                # Create centroids
                zones = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 2611, 2612, 2613, 2614, 2616, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 3000, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008]
                zonesOD = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 2611, 2612, 2613, 2614, 2616, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 3000]
                zonesODTransit = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 2612, 2613, 2616, 2622, 2623, 2624, 2625, 2628, 2629, 3000, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4008]


                data = [1001, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                writer.writerow(data)
                data = [1002, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                writer.writerow(data)
                data = [1003, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                writer.writerow(data)
                data = [1004, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                writer.writerow(data)
                data = [1005, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                writer.writerow(data)
                data = [1006, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                writer.writerow(data)
                data = [1007, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                writer.writerow(data)
                data = [2611, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                writer.writerow(data)
                data = [2612, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                writer.writerow(data)
                data = [2613, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                writer.writerow(data)
                data = [2614, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                writer.writerow(data)
                data = [2616, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                writer.writerow(data)
                data = [2622, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                writer.writerow(data)
                data = [2623, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                writer.writerow(data)
                data = [2624, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                writer.writerow(data)
                data = [2625, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                writer.writerow(data)
                data = [2626, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                writer.writerow(data)
                data = [2627, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                writer.writerow(data)
                data = [2628, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                writer.writerow(data)
                data = [2629, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                writer.writerow(data)
                data = [3000, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                writer.writerow(data)

                data = [4000, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                writer.writerow(data)
                data = [4001, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                writer.writerow(data)
                data = [4002, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                writer.writerow(data)
                data = [4003, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                writer.writerow(data)
                data = [4004, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                writer.writerow(data)
                data = [4005, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                writer.writerow(data)
                data = [4006, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                writer.writerow(data)
                data = [4007, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                writer.writerow(data)
                data = [4008, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                writer.writerow(data)

                if 'Multimodal' in networkName:

                    # Create nodes neutral layer
                    data = [21001, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                    writer.writerow(data)
                    data = [21002, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                    writer.writerow(data)
                    data = [21003, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                    writer.writerow(data)
                    data = [21004, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                    writer.writerow(data)
                    data = [21005, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                    writer.writerow(data)
                    data = [21006, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                    writer.writerow(data)
                    data = [21007, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                    writer.writerow(data)
                    data = [22611, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                    writer.writerow(data)
                    data = [22612, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                    writer.writerow(data)
                    data = [22613, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                    writer.writerow(data)
                    data = [22614, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                    writer.writerow(data)
                    data = [22616, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                    writer.writerow(data)
                    data = [22622, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                    writer.writerow(data)
                    data = [22623, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                    writer.writerow(data)
                    data = [22624, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                    writer.writerow(data)
                    data = [22625, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                    writer.writerow(data)
                    data = [22626, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                    writer.writerow(data)
                    data = [22627, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                    writer.writerow(data)
                    data = [22628, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                    writer.writerow(data)
                    data = [22629, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                    writer.writerow(data)
                    data = [23000, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                    writer.writerow(data)

                    data = [24000, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                    writer.writerow(data)
                    data = [24001, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                    writer.writerow(data)
                    data = [24002, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                    writer.writerow(data)
                    data = [24003, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                    writer.writerow(data)
                    data = [24004, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                    writer.writerow(data)
                    data = [24005, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                    writer.writerow(data)
                    data = [24006, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                    writer.writerow(data)
                    data = [24007, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                    writer.writerow(data)
                    data = [24008, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                    writer.writerow(data)

                layerNumber = [30000, 40000, 50000, 60000, 70000, 80000]
                locations = [zones, zones, zones, zones, zones, zones]
                

                # zonesFrom = [2626, 2626, 2622, 2622, 2625, 2625, 2625, 1004, 1004, 1003, 1003, 1002, 1002, 1001, 1001, 2616, 2616, 1007, 1007, 1006, 1006, 1006, 1005, 2629, 2629, 2629, 2629, 2623, 2623, 2624, 2624, 4005, 4005, 4005, 2614, 2614, 2614, 4000, 4000, 4001, 4001, 4001, 2611, 2611, 2611, 2611, 2612, 2612, 2628, 2628, 4004, 4004, 4004, 3000, 3000, 2627, 2627]
                # zonesTo = [2623, 2622, 2623, 2625, 2624, 4006, 1004, 4006, 1003, 4000, 1002, 2613, 1001, 4002, 2616, 2612, 1007, 4003, 1006, 2628, 2629, 1005, 2629, 2628, 4008, 2627, 2623, 2627, 2624, 4007, 4005, 4006, 2613, 3000, 4006, 4000, 2613, 4001, 2613, 4002, 2611, 2613, 4002, 4003, 4004, 3000, 4002, 4003, 4003, 4008, 4003, 4008, 3000, 2613, 4007, 4007, 4008]
                # # For transit layer (fewer connections)
                # zonesFromTransit = [1002, 4000, 4000, 4000, 4001, 4001, 4001, 4002, 4002, 2611, 2611, 2611, 4003, 4003, 4004, 4004, 2628, 2628, 1006, 4008, 2629, 2623, 2623, 2623, 2622, 2625, 2625, 4006, 4006, 4005, 4005, 4005, 2614, 2613, 3000, 4007, 4007]
                # zonesToTransit = [4000, 4001, 2613, 2614, 4002, 2611, 2613, 1001, 2611, 4003, 4004, 3000, 2628, 4004, 4008, 3000, 1006, 4008, 1005, 2629, 2623, 2627, 2624, 2622, 2625, 2624, 4006, 4005, 2614, 2613, 3000, 2624, 2613, 3000, 4007, 2624, 2627]

                # distance = [43, 38, 35, 43, 27, 14, 45, 39, 55, 49, 47, 49, 46, 48, 63, 43, 73, 61, 34, 31, 55, 96, 51, 49, 50, 38, 42, 31, 42, 9, 18, 27, 23, 18, 25, 24, 22, 22, 24, 17, 18, 18, 25, 22, 19, 20, 15, 34, 31, 8, 13, 24, 19, 15, 31, 16, 27]
                # distance = [i/24*1 for i in distance]  # Scaling: distance = distance / 24, assumed 50% extra length due to aggregation links and 'hemelsbrede afstand', based on distances in OmniTRANS when selection OD-pairs.

                # distanceTransit = distance
                
                zonesFrom = [1001,    1001,    1002,    1002,    1003,    1003,    1004,    1004,    1005,    1006,    1006,    1006,    1007,    1007,    2611,    2611,    2611,    2611,    2612,    2612,    2614,    2614,    2614,    2616,    2616,    2622,    2622,    2623,    2623,    2624,    2624,    2625,    2625,    2625,    2626,    2626,    2627,    2627,    2628,    2628,    2629,    2629,    2629,    2629,    3000,    3000,    4000,    4000,    4001,    4001,    4001,    4004,    4004,    4004,    4005,    4005,    4005]
                zonesTo   = [4002,    2616,    2613,    1001,    4000,    1002,    4006,    1003,    2629,    2628,    2629,    1005,    4003,    1006,    4002,    4003,    4004,    3000,    4002,    4003,    4006,    4000,    2613,    2612,    1007,    2623,    2625,    2627,    2624,    4007,    4005,    2624,    4006,    1004,    2623,    2622,    4007,    4008,    4003,    4008,    2628,    4008,    2627,    2623,    2613,    4007,    4001,    2613,    4002,    2611,    2613,    4003,    4008,    3000,    4006,    2613,    3000]
                distance  = [2.04,    2.04,    4.28,    2.35,    5.54,    2.82,    3.81,    4.20,    6.24,    3.61,    3.61,    3.61,    3.61,    3.61,    2.33,    0.93,    0.93,    0.93,    1.36,    2.33,    2.47,    1.38,    1.52,    3.61,    2.04,    2.62,    2.37,    4.15,    2.80,    2.64,    2.19,    2.19,    2.47,    4.67,    5.57,    3.15,    2.64,    2.95,    2.54,    2.54,    1.99,    6.24,    4.52,    4.52,    0.93,    2.64,    1.52,    1.68,    1.36,    0.93,    1.68,    2.54,    2.54,    1.52,    1.38,    1.68,    1.52]

                # # For transit layer (fewer connections)                                                                                                                                                                                                                    
                # zonesFromTransit = [1002,    1006,    2611,    2611,    2611,    2613,    2614,    2622,    2623,    2623,    2623,    2625,    2625,    2628,    2628,    2629,    3000,    4000,    4000,    4000,    4001,    4001,    4001,    4002,    4002,    4003,    4003,    4004,    4004,    4005,    4005,    4005,    4006,    4006,    4007,    4007,    4008]                                                                                
                # zonesToTransit   = [4000,    1005,    3000,    4003,    4004,    3000,    2613,    2625,    2622,    2624,    2627,    2624,    4006,    1006,    4008,    2623,    4007,    2613,    2614,    4001,    2611,    2613,    4002,    1001,    2611,    2628,    4004,    3000,    4008,    2613,    2624,    3000,    2614,    4005,    2624,    2627,    2629]                                                                               
                # distanceTransit  = [5.54,    3.61,    0.93,    2.33,    0.93,    0.93,    1.52,    2.37,    2.62,    2.80,    4.15,    2.19,    2.47,    3.61,    2.54,    4.52,    2.64,    1.68,    1.38,    1.52,    0.93,    1.68,    1.36,    2.04,    2.33,    2.54,    2.54,    1.52,    2.54,    1.68,    2.19,    1.52,    2.47,    1.38,    2.64,    2.64,    6.24]                                                                                

                # For carpool layer (fewer connection and only to Delft station (3000))
                # zonesFromCarpool = [1001,    1002,    1003,    1004,    1006,    1007,    2611,    2612,    2613,    2614,    2616,    2622,    2623,    2624,    2625,    2626,    2627,    2628,    2629,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000]
                # zonesToCarpool   = [3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    3000,    1001,    1002,    1003,    1004,    1006,    1007,    2611,    2612,    2613,    2614,    2616,    2622,    2623,    2624,    2625,    2626,    2627,    2628,    2629]
                # distanceCarpool  = [3.5,     3.6,     5.6,     4.1,     4.6,     5.2,     2.2,     2.2,     1.5,     2.9,     4.7,     4.6,     3.7,     2.6,     3.2,     6.4,     2.5,     3.6,     4.9,     3.5,     3.6,     5.6,     4.1,     4.6,     5.2,     2.2,     2.2,     1.5,     2.9,     4.7,     4.6,     3.7,     2.6,     3.2,     6.4,     2.5,     3.6,     4.9]

                resistance = [6, 6, 6, 6, 6, 6]
                resistanceToNeutral = [1, 1, 1, 1, 1, 1]
                neutralToResistance = [1, 1, 1, 1, 1, 1]
                distanceNeutralToMode = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
                capacity = [2000, 2000, 2000, 2000, 2000, 2000]
                speed = [34.668441679929636, 30.67680672911914, 29.074451916316153, 14.621403427954009, 5, 10] # From ODIN 2019
                defSpeed = [34.668441679929636, 30.67680672911914, 29.074451916316153, 14.621403427954009, 5, 10] # From ODIN 2019
                mode = ['Car', 'Pool', 'Transit', 'Bicycle', 'Walk', 'Future'] # Pool = carpool, coding reasons, just called pool.

                for j in range(nrOfModes):

                    if (j != 2):
                        # Create nodes
                        data = [layerNumber[j]+1001, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        # Create links
                        for k in range(len(zonesFrom)):
                            data = [layerNumber[j]+k, 'link', 0, 0, layerNumber[j]+zonesFrom[k], layerNumber[j]+zonesTo[k], resistance[j], capacity[j], distance[k], speed[j], defSpeed[j], mode[j]]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000, 'link', 0, 0, layerNumber[j]+zonesTo[k], layerNumber[j]+zonesFrom[k], resistance[j], capacity[j], distance[k], speed[j], defSpeed[j], mode[j]]
                            writer.writerow(data)

                        if 'Multimodal' in networkName:
                            for i in zones:
                                # One-way
                                data = [layerNumber[j]+i, 'link', 0, 0, layerNumber[j]+i, neutralLayerFactor+i, resistance[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                                writer.writerow(data)
                                # Other-way
                                data = [layerNumber[j]+i+5000, 'link', 0, 0, neutralLayerFactor+i, layerNumber[j]+i, resistance[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                                writer.writerow(data)
                        else:
                            for i in zonesOD:
                                # One-way
                                data = [layerNumber[j]+i, 'link', 0, 0, layerNumber[j]+i, 0+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                                writer.writerow(data)
                                # Other-way
                                data = [layerNumber[j]+i+5000, 'link', 0, 0, 0+i, layerNumber[j]+i, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                                writer.writerow(data)


                    elif j == 2: # Transit layer

                        # Steps
                        # Create nodes, basic transit layer
                        # Create array with all edges of each line
                        # Create nodes for each line
                        # Create edges between line-layer and main transit layer

                        # Create nodes, basic transit layer
                        data = [layerNumber[j]+1001, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        # Create links to neutral layer
                        for i in zonesODTransit:
                            # One-way
                            data = [layerNumber[j]+i, 'link', 0, 0, layerNumber[j]+i, neutralLayerFactor+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], mode[j] + 'ToNeutral']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000, 'link', 0, 0, neutralLayerFactor+i, layerNumber[j]+i, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speed[j], defSpeed[j], 'NeutralTo' + mode[j]]
                            writer.writerow(data)

                        ##########################################################################################################################################


                        # Tram 1
                        transitLine      = 'Tram001'
                        transitNumber    = 100000
                        speedTransit     = 19 # [km/h]

                        nodesTransit     = [2622, 2625, 2624, 4005, 3000, 4001, 4002, 1001]
                        zonesFromTransit = [2622, 2625, 2624, 4005, 3000, 4001, 4002]
                        zonesToTransit   = [2625, 2624, 4005, 3000, 4001, 4002, 1001]
                        distanceTransit  = [2.37, 2.19, 2.19, 1.52, 1.68, 1.36, 2.04]


                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)


                        # Tram 19
                        transitLine      = 'Tram019'
                        transitNumber    = 1900000
                        speedTransit     = 19 # [km/h]

                        nodesTransit     = [3000, 4001, 4002, 2616]
                        zonesFromTransit = [3000, 4001, 4002]
                        zonesToTransit   = [4001, 4002, 2616]
                        distanceTransit  = [1.68, 1.36, 2.04]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)



                        # Bus 455
                        transitLine      = 'Bus455'
                        transitNumber    = 45500000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [1004, 4006, 4005, 3000, 4004, 4008, 2628, 1006, 1007]
                        zonesFromTransit = [1004, 4006, 4005, 3000, 4004, 4008, 2628, 1006]
                        zonesToTransit   = [4006, 4005, 3000, 4004, 4008, 2628, 1006, 1007]
                        distanceTransit  = [3.81, 1.38, 1.52, 1.52, 2.54, 2.54, 3.61, 3.61]


                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 40
                        transitLine      = 'Bus040'
                        transitNumber    = 4000000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4004, 4008, 2629, 1005]
                        zonesFromTransit = [3000, 4004, 4008, 2629]
                        zonesToTransit   = [4004, 4008, 2629, 1005]
                        distanceTransit  = [1.52, 2.54, 6.24, 6.24]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 174
                        transitLine      = 'Bus174'
                        transitNumber    = 17400000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4004, 4008, 2628, 1006, 1007]
                        zonesFromTransit = [3000, 4004, 4008, 2628, 1006]
                        zonesToTransit   = [4004, 4008, 2628, 1006, 1007]
                        distanceTransit  = [1.52, 2.54, 2.54, 3.61, 3.61]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 60
                        transitLine      = 'Bus060'
                        transitNumber    = 6000000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4004, 4003, 2612, 2616]
                        zonesFromTransit = [3000, 4004, 4003, 2612]
                        zonesToTransit   = [4004, 4003, 2612, 2616]
                        distanceTransit  = [1.52, 2.54, 2.33, 3.61]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 61
                        transitLine      = 'Bus061'
                        transitNumber    = 6100000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 2613, 4000, 1002]
                        zonesFromTransit = [3000, 2613, 4000]
                        zonesToTransit   = [2613, 4000, 1002]
                        distanceTransit  = [0.93, 1.68, 4.28]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 62
                        transitLine      = 'Bus062'
                        transitNumber    = 6200000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4004, 4003, 1006]
                        zonesFromTransit = [3000, 4004, 4003]
                        zonesToTransit   = [4004, 4003, 1006]
                        distanceTransit  = [1.52, 2.54, 3.61]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 63
                        transitLine      = 'Bus063'
                        transitNumber    = 6300000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4004, 4003]
                        zonesFromTransit = [3000, 4004]
                        zonesToTransit   = [4004, 4003]
                        distanceTransit  = [1.52, 2.54]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 64
                        transitLine      = 'Bus064'
                        transitNumber    = 6400000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4005, 2624, 2623, 2622]
                        zonesFromTransit = [3000, 4005, 2624, 2623]
                        zonesToTransit   = [4005, 2624, 2623, 2622]
                        distanceTransit  = [1.52, 2.19, 2.80, 2.62]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 69
                        transitLine      = 'Bus069'
                        transitNumber    = 6900000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4004, 4008, 2629]
                        zonesFromTransit = [3000, 4004, 4008]
                        zonesToTransit   = [4004, 4008, 2629]
                        distanceTransit  = [1.52, 2.54, 6.24]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 33
                        transitLine      = 'Bus033'
                        transitNumber    = 3300000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4005, 4006, 1004]
                        zonesFromTransit = [3000, 4005, 4006]
                        zonesToTransit   = [4005, 4006, 1004]
                        distanceTransit  = [1.52, 1.38, 3.81]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 37
                        transitLine      = 'Bus037'
                        transitNumber    = 3700000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4005, 4006, 1003]
                        zonesFromTransit = [3000, 4005, 4006]
                        zonesToTransit   = [4005, 4006, 1003]
                        distanceTransit  = [1.52, 1.38, 6.10]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Bus 53
                        transitLine      = 'Bus053'
                        transitNumber    = 5300000
                        speedTransit     = 25 # [km/h]

                        nodesTransit     = [3000, 4001, 4002, 1001]
                        zonesFromTransit = [3000, 4001, 4002]
                        zonesToTransit   = [4001, 4002, 1001]
                        distanceTransit  = [1.68, 1.36, 2.04]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)

                        # Train
                        transitLine      = 'Train015'
                        transitNumber    = 1500000
                        speedTransit     = 42 # [km/h]

                        nodesTransit     = [3000, 4002, 1001]
                        zonesFromTransit = [3000]
                        zonesToTransit   = [4002]
                        distanceTransit  = [2.10]

                        data = [layerNumber[j]+1001+transitNumber, 'node', 83514.38281250000000, 450414.2500000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1002+transitNumber, 'node', 81754.23437500000000, 450006.8125000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1003+transitNumber, 'node', 80740.95312500000000, 448160.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1004+transitNumber, 'node', 81413.66406250000000, 445168.9375000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1005+transitNumber, 'node', 86561.98573592488537, 442724.2360644022119, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1006+transitNumber, 'node', 86889.34375000000000, 446489.3437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+1007+transitNumber, 'node', 87433.92968750000000, 447491.8437500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2611+transitNumber, 'node', (84290.72656250000000 + 84582.06250000000000)/2, (447866.0625000000000 + 447480.3437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2612+transitNumber, 'node', (84100.17968750000000 + 85027.17187500000000)/2, (449364.6562500000000 + 448265.3125000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2613+transitNumber, 'node', 83649.67968750000000, 447266.4687500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2614+transitNumber, 'node', 82895.68750000000000, 447039.6562500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2616+transitNumber, 'node', 85612.89337839159998, 449648.7369987610145, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2622+transitNumber, 'node', 83076.39417873916681, 443848.4507972683059, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2623+transitNumber, 'node', 84616.39843750000000, 444239.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2624+transitNumber, 'node', 84268.64843750000000, 445737.0000000000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2625+transitNumber, 'node', 83044.39843750000000, 445531.5312500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2626+transitNumber, 'node', 83598.14237760189280, 442713.5747573578846, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2627+transitNumber, 'node', (84720.35156250000000 + 85360.65625000000000)/2, (446023.1250000000000 + 444415.7500000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2628+transitNumber, 'node', (85991.51634928799467 + 85631.52343750000000)/2, (446364.2944520757882 + 446012.8437500000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+2629+transitNumber, 'node', 86073.45312500000000, 444506.9062500000000, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+3000+transitNumber, 'node', 84220.19752310098556, 447017.0427329398808, 0, 0]
                        writer.writerow(data)

                        data = [layerNumber[j]+4000+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.448 + 80740.95312500000000*0.552), ((447866.0625000000000 + 447480.3437500000000)/2*0.448 + 448160.9062500000000*0.552), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4001+transitNumber, 'node', (81754.23437500000000 + (85991.51634928799467 + 85631.52343750000000)/2)/2, (450006.8125000000000 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4002+transitNumber, 'node', (83514.38281250000000*0.35 + (84290.72656250000000 + 84582.06250000000000)/2*0.65), (450414.2500000000000*0.35 + (447866.0625000000000 + 447480.3437500000000)/2*0.65), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4003+transitNumber, 'node', ((84100.17968750000000 + 85027.17187500000000)/2 + (85991.51634928799467 + 85631.52343750000000)/2)/2, ((449364.6562500000000 + 448265.3125000000000)/2 + (446364.2944520757882 + 446012.8437500000000)/2)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4004+transitNumber, 'node', ((84290.72656250000000 + 84582.06250000000000)/2*0.59 + (85991.51634928799467 + 85631.52343750000000)/2*0.41), ((447866.0625000000000 + 447480.3437500000000)/2*0.59 + (446364.2944520757882 + 446012.8437500000000)/2*0.41), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4005+transitNumber, 'node', (84220.19752310098556 + 84268.64843750000000)/2, (447017.0427329398808 + 445737.0000000000000)/2, 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4006+transitNumber, 'node', (82895.68750000000000*0.37 + 83044.39843750000000*0.63), (447039.6562500000000*0.37 + 445531.5312500000000*0.63), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4007+transitNumber, 'node', (84268.64843750000000*0.86 + 86889.34375000000000*0.14), (445737.0000000000000*0.86 + 446489.3437500000000*0.14), 0, 0]
                        writer.writerow(data)
                        data = [layerNumber[j]+4008+transitNumber, 'node', ((85991.51634928799467 + 85631.52343750000000)/2*0.89 + (82895.68750000000000*0.37 + 83044.39843750000000*0.63)*0.11), ((446364.2944520757882 + 446012.8437500000000)/2*0.89 + (447039.6562500000000*0.37 + 445531.5312500000000*0.63)*0.11), 0, 0]
                        writer.writerow(data)

                        for k in range(len(zonesFromTransit)):
                            data = [layerNumber[j]+k+transitNumber, 'link', 0, 0, layerNumber[j]+zonesFromTransit[k]+transitNumber, layerNumber[j]+zonesToTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)
                            data = [layerNumber[j]+k+5000+transitNumber, 'link', 0, 0, layerNumber[j]+zonesToTransit[k]+transitNumber, layerNumber[j]+zonesFromTransit[k]+transitNumber, resistance[j], capacity[j], distanceTransit[k], speedTransit, speedTransit, transitLine]
                            writer.writerow(data)

                        for i in nodesTransit:
                            # One-way
                            data = [layerNumber[j]+i+transitNumber, 'link', 0, 0, layerNumber[j]+i+transitNumber, layerNumber[j]+i, resistanceToNeutral[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, transitLine + 'ToTransit']
                            writer.writerow(data)
                            # Other-way
                            data = [layerNumber[j]+i+5000+transitNumber, 'link', 0, 0, layerNumber[j]+i, layerNumber[j]+i+transitNumber, neutralToResistance[j], capacity[j], distanceNeutralToMode[j], speedTransit, speedTransit, 'TransitTo' + transitLine]
                            writer.writerow(data)


                if 'Multimodal' in networkName:
                    # OD centroids to neutral layer
                    for i in zonesOD:
                        # One-way
                        data = [10000+i, 'link', 0, 0, i, 20000+i, 5, 2000, 0.001, 50, 50, 'ODToNeutral']
                        writer.writerow(data)
                        # Other-way
                        data = [10000+i+5000, 'link', 0, 0, 20000+i, i, 5, 2000, 0.001, 50, 50, 'NeutralToOD']
                        writer.writerow(data)

######################################################################################################################################################################################################

    # CSV to JSON
    
    entriesFile = r'data/simpleNetwork.csv' # Path to csv

    # Create an empty dictionary to collect values
    nodes = []
    nodesTemp = {99999: {'id': '99999', 'x': 0, 'y': 0}}
    links = []

    with open(entriesFile, encoding='utf-8') as csvFile:
        csvContent = csv.DictReader(csvFile)

        entries = []

        # Add all nodes in dictionary
        for row in csvContent:
            # Assign the values in the row to the appropriate key
            if (row["type"] == 'node'):
                dictNode = {
                    "id": row['name'],
                    "type": row['type'],
                    "pos": [ast.literal_eval(row['pos_x']), ast.literal_eval(row['pos_y'])]
                }
                nodes.append(dictNode)
                nodesTemp[row['name']] = {'id': row['name'], 'x': ast.literal_eval(row['pos_x']), 'y': ast.literal_eval(row['pos_y'])}

    with open(entriesFile, encoding='utf-8') as csvFile:
        csvContent = csv.DictReader(csvFile)
        # Add all links in dictionary, in seperate loop from adding nodes, since nodes dict is used for calculation of distance in dictLink.
        for row in csvContent:
            # Assign the values in the row to the appropriate key
            if (row["type"] == 'link'):

                # Calculate length of link based on locations (in km)
                distance = ast.literal_eval(row['distance'])
                # if (row['source'][-4:] != row['target'][-4:]) and ('Neutral' not in row['mode']):
                #     distance = ((((nodesTemp[row['source']]['x'] - nodesTemp[row['target']]['x'])/1000)**2 + ((nodesTemp[row['source']]['y'] - nodesTemp[row['target']]['y'])/1000)**2)**0.5)/1000

                dictLink = {
                    "source": row['source'],
                    "target": row['target'],
                    "resistance10": ast.literal_eval(row['resistance']),
                    "resistance11": ast.literal_eval(row['resistance']),
                    "resistance12": ast.literal_eval(row['resistance']),
                    "resistance13": ast.literal_eval(row['resistance']),
                    "resistance14": ast.literal_eval(row['resistance']),
                    "resistance15": ast.literal_eval(row['resistance']),
                    "resistance20": ast.literal_eval(row['resistance']),
                    "resistance21": ast.literal_eval(row['resistance']),
                    "resistance22": ast.literal_eval(row['resistance']),
                    "resistance23": ast.literal_eval(row['resistance']),
                    "resistance24": ast.literal_eval(row['resistance']),
                    "resistance25": ast.literal_eval(row['resistance']),
                    "capacity": ast.literal_eval(row['capacity']),
                    "distance": distance,
                    "speed": ast.literal_eval(row['speed']),
                    "defSpeed": ast.literal_eval(row['defSpeed']),
                    "mode": row['mode'],
                }
                links.append(dictLink)

        dictNodes = {
            "nodes": nodes,
            "links": links,
            "graph": {},
            "directed": 'False',
            "multigraph": 'False'
        }

    with open('data/simpleNetwork.json', 'w', encoding='utf-8') as f:
        json.dump(dictNodes, f, ensure_ascii=False, indent=4)

    print('Finished converting CSV to JSON')
    return links


def calcFirstPositions(trips_timeStep, G, networkName, seed, MNLbeta, MNLmu, pos):

    trips = []
    linksCongestion = []

    for index, row in trips_timeStep.iterrows():

        # Determine next node to reach goal
        prevPrevSource = "NAN" # There is no previous source yet
        prevSource = "NAN" # There is no previous source yet
        recSource = str(int(row["origin"]))
        recTarget = str(int(row["destination"]))
        cluster = str(round(row["type"]))
        nextTarget = nextNode(G, prevPrevSource, prevSource, recSource, recTarget, cluster, networkName, seed, MNLbeta, MNLmu)

        # Initialize starting point in network
        positionLink = 0
        pos_x = pos[recSource][0]
        pos_y = pos[recSource][1]
        speed = 0

        # Write in dataset to be called in next step as start
        dictTrip = {
            "id": str(round(float(row["id"]), 1)),
            "prevPrevSource": str(row["origin"]),
            "prevSource": str(int(row["origin"])),
            "recSource": recSource,
            "recTarget": recTarget,
            "nextTarget": nextTarget,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "positionLink": positionLink,
            "speed": speed,
            "cluster": str(row["type"]),
        }
        trips.append(dictTrip)

        # Write agents on link for congestion calculation
        dictLinks = {
            "recSource": recSource,
            "recTarget": nextTarget,
        }
        linksCongestion.append(dictLinks)

    return [trips, linksCongestion]


def calcNextPositions(tripsPrev, G, timeStep, networkName, seed, MNLbeta, MNLmu, pos):

    trips = []
    linksCongestion = []

    # Calculate next positions trips in network
    for index, row in tripsPrev.iterrows():

        # Determine next node to reach goal
        prevPrevSource = str(row["prevPrevSource"])
        prevSource = str(row["prevSource"])
        recSource = str(row["recSource"])
        recTarget = str(row["recTarget"])
        nextTarget = str(row["nextTarget"])
        cluster = str(row["cluster"])

        # Speed is calculated when entering link.
        # When link gets more congested, speed goed down for subsequent agents, but stays the same for agents already on the link.
        # When link gets less congested, speed goes up for all agents on the link if this speed is higher than their initial speed on this link.
        speed = max(G[recSource][nextTarget][0]['speed'], row["speed"])

        # Calculate location (coordinates, not nodes) next timestep
        # Value between 0 and 1, depending on how far the agent is progressed on one link
        if (G[recSource][nextTarget][0]['distance'] != 0):
            positionLink = timeStep * speed / G[recSource][nextTarget][0]['distance'] + row["positionLink"]
        else:
            positionLink = 1 # Move to next link, since distance = 0 and evaluate again
        
        # Update recSource if end of link has been reached
        if positionLink >= 1:
            # try:
            if nextTarget != recTarget:
                recSource = str(row["nextTarget"])
                prevPrevSource = str(row["prevSource"])
                prevSource = str(row["recSource"])
                nextTarget = nextNode(G, prevPrevSource, prevSource, recSource, recTarget, cluster, networkName, seed, MNLbeta, MNLmu)
            else: # End of route, agent is removed from network
                continue

            # Calculate proportion in new link during timestep
            proportion = 1 - ((1 - row["positionLink"]) / (positionLink - row["positionLink"]))
            speed = G[recSource][nextTarget][0]['speed']
            positionLink = proportion * timeStep * speed / G[recSource][nextTarget][0]['distance']


        pos_x = pos[recSource][0] * (1 - positionLink) + pos[nextTarget][0] * positionLink
        pos_y = pos[recSource][1] * (1 - positionLink) + pos[nextTarget][1] * positionLink

        # Write in dataset to be called in next step as start (to calculate congestion)
        dictTrip = {
            "id": str(round(float(row["id"]), 1)),
            "prevPrevSource": prevPrevSource, # prevPrevSource
            "prevSource": prevSource, # prevSource,
            "recSource": recSource,
            "recTarget": recTarget,
            "nextTarget": nextTarget,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "positionLink": positionLink,
            "speed": speed,
            "cluster": str(row["cluster"]),
        }
        trips.append(dictTrip)

        # Write agents on link for congestion calculation
        dictLinks = {
            "recSource": recSource,
            "recTarget": nextTarget,
        }
        linksCongestion.append(dictLinks)

    return [trips, linksCongestion]


def simulateNetwork(futureCharRow, links, updateResInt, scalingSampleSize, time, timeStep, networkName, seed, MNLbeta, MNLmu, nameFuture):

    # Load json networkx datafile
    with open('data/simpleNetwork.json','r') as infile:
        G = nx.json_graph.node_link_graph(json.load(infile))

    # Load link capacities
    # dfBPR = pd.read_csv('dataset/BPR_all_links.csv', sep=';', decimal=",")

    for i in time:
        if i > 0:
            G = nx.json_graph.node_link_graph(dictNodes)
        pos = nx.get_node_attributes(G, 'pos')

        # Simulate
        newTrips = pd.read_json('data/trips.json')

        # Initialize shortestPathsList
        shortestPathsList = np.array([0, 0])

        # Create as many processes as there are CPUs on your machine
        num_processes = mp.cpu_count() - 1 # 8 - 1 to account for last chunk if range is not covering all trips exactly

        # Initialize trips dictionaries
        trips = []
        linksCongestion = []
        if i > 0:
            tripsPrev = pd.read_pickle('data/tripsTimeStep' + str(i-1) + '.pkl') # Load trips previous timestep
            if len(tripsPrev.index) > 0:
                if len(tripsPrev.index) <= 2*num_processes:
                    num_processes = 1
                # Calculate chunk size and divide trips into chunks to be iterated through
                chunk_size = int(len(tripsPrev.index)/(num_processes))
                chunks = [tripsPrev[n:min(n + chunk_size, tripsPrev.shape[0])] for n in range(0, tripsPrev.shape[0], chunk_size)]

                # Calculate next positions trips in network for chunks parallel
                with mp.Pool() as pool:
                    pool = mp.Pool(processes=num_processes)
                    async_results = [pool.apply_async(calcNextPositions, args=(chunks[m], G, timeStep, networkName, seed, MNLbeta, MNLmu, pos)) for m in range(num_processes)]
                    results = [ar.get() for ar in async_results]
                    pool.close()
                    pool.join()

                for m in range(num_processes):
                    tripsTemp = results[m][0]
                    linksCongestionTemp = results[m][1]
                    trips.extend(tripsTemp)
                    linksCongestion.extend(linksCongestionTemp)
            
        # Load new trips in network
        trips_timeStep = newTrips.loc[newTrips["depTime"] == i]
        if len(trips_timeStep) > 0:
            if len(trips_timeStep.index) <= 2*num_processes:
                    num_processes = 1
                    chunks = [trips_timeStep]
            else:
                # Calculate chunk size and divide trips into chunks to be iterated through
                chunk_size = int(len(trips_timeStep.index)/(num_processes))
                chunks = [trips_timeStep[n:min(n + chunk_size, trips_timeStep.shape[0])] for n in range(0, trips_timeStep.shape[0], chunk_size)]

            # Calculate next positions trips in network for chunks parallel
            with mp.Pool() as pool:
                pool = mp.Pool(processes=num_processes)
                async_results = [pool.apply_async(calcFirstPositions, args=(chunks[m], G, networkName, seed, MNLbeta, MNLmu, pos)) for m in range(num_processes)]
                results = [ar.get() for ar in async_results]
                pool.close()
                pool.join()

            for m in range(num_processes):
                tripsTemp = results[m][0]
                linksCongestionTemp = results[m][1]
                trips.extend(tripsTemp)
                linksCongestion.extend(linksCongestionTemp)

        # Save all trips with details for current timestep
        CWD = Path(__file__).parent
        filename = str(CWD) + '/data/tripsTimeStep' + str(i) + '.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        tripsDF = pd.DataFrame.from_dict(trips)
        tripsDF.to_pickle(filename)

        # Determine total agents on each link for congestion calculation and write in network
        nodes = list(G.nodes(data=True))
        congestion = list(G.edges(data=True))

        dictNodes = []
        dictLinks = []

        # Set congestion to 0
        for row in nodes:
            try:
                dictNode = {
                    "id": row[0],
                    "type": row[1]['type'],
                    "pos": [row[1]['pos'][0], row[1]['pos'][1]]
                }
                dictNodes.append(dictNode)
            except:
                print('FAILED FOR', row)
                pass 


        # Set resistance (i.e. # agents count) to 0
        for count, value in enumerate(congestion):
            congestion[count][2]["resistance"] = 0

        # Update congestion changed links
        # PCU values: car, carpool = 1.0, bicycle = 0.2. Transit assumed to be trams, so on different network.
        for j in linksCongestion:
            for c, k in enumerate(congestion):
                # Calculate congestion for each mode with shared infrastructure

                # Only for car layer
                if (int(3) == int(str(j["recSource"])[0])) and (int(3) == int(str(j["recTarget"])[0])):
                    # Add agents on current layer
                    if (int(k[0]) == int(j["recSource"])) and (int(k[1]) == int(j["recTarget"])):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on carpool-layer
                    if (int(k[0]) == int(j["recSource"])+10000) and (int(k[1]) == int(j["recTarget"])+10000):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on bicycle-layer
                    if (int(k[0]) == int(j["recSource"])+30000) and (int(k[1]) == int(j["recTarget"])+30000):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on future-layer
                    if (int(k[0]) == int(j["recSource"])+50000) and (int(k[1]) == int(j["recTarget"])+50000):
                        congestion[c][2]["resistance"] += 1 # PCU

                # Only for carpool layer
                if (int(4) == int(str(j["recSource"])[0])) and (int(4) == int(str(j["recTarget"])[0])):
                    # Add agents on current layer
                    if (int(k[0]) == int(j["recSource"])) and (int(k[1]) == int(j["recTarget"])):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on car-layer
                    if (int(k[0]) == int(j["recSource"])-10000) and (int(k[1]) == int(j["recTarget"])-10000):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on bicycle-layer
                    if (int(k[0]) == int(j["recSource"])+20000) and (int(k[1]) == int(j["recTarget"])+20000):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on future-layer
                    if (int(k[0]) == int(j["recSource"])+40000) and (int(k[1]) == int(j["recTarget"])+40000):
                        congestion[c][2]["resistance"] += 1 # PCU

                # Only for bicycle layer
                if (int(6) == int(str(j["recSource"])[0])) and (int(6) == int(str(j["recTarget"])[0])):
                    # Add agents on current layer
                    if (int(k[0]) == int(j["recSource"])) and (int(k[1]) == int(j["recTarget"])):
                        congestion[c][2]["resistance"] += 0.2 # PCU
                    # Add agents on car-layer
                    if (int(k[0]) == int(j["recSource"])-30000) and (int(k[1]) == int(j["recTarget"])-30000):
                        congestion[c][2]["resistance"] += 0.2 # PCU
                    # Add agents on carpool-layer
                    if (int(k[0]) == int(j["recSource"])-20000) and (int(k[1]) == int(j["recTarget"])-20000):
                        congestion[c][2]["resistance"] += 0.2 # PCU
                    # Add agents on future-layer
                    if (int(k[0]) == int(j["recSource"])+20000) and (int(k[1]) == int(j["recTarget"])+20000):
                        congestion[c][2]["resistance"] += 0.2 # PCU

                # Only for future - AV layer
                if ((nameFuture == "Shared Autonomous Car") and int(8) == int(str(j["recSource"])[0])) and (int(8) == int(str(j["recTarget"])[0])):
                    # Add agents on current layer
                    if (int(k[0]) == int(j["recSource"])) and (int(k[1]) == int(j["recTarget"])):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on car-layer
                    if (int(k[0]) == int(j["recSource"])-50000) and (int(k[1]) == int(j["recTarget"])-50000):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on carpool-layer
                    if (int(k[0]) == int(j["recSource"])-40000) and (int(k[1]) == int(j["recTarget"])-40000):
                        congestion[c][2]["resistance"] += 1 # PCU
                    # Add agents on bicycle-layer
                    if (int(k[0]) == int(j["recSource"])-20000) and (int(k[1]) == int(j["recTarget"])-20000):
                        congestion[c][2]["resistance"] += 1 # PCU

                # Only for future - electric step layer
                if ((nameFuture == "Electric Step") and int(8) == int(str(j["recSource"])[0])) and (int(8) == int(str(j["recTarget"])[0])):
                    # Add agents on current layer
                    if (int(k[0]) == int(j["recSource"])) and (int(k[1]) == int(j["recTarget"])):
                        congestion[c][2]["resistance"] += 0.2 # PCU
                    # Add agents on car-layer
                    if (int(k[0]) == int(j["recSource"])-50000) and (int(k[1]) == int(j["recTarget"])-50000):
                        congestion[c][2]["resistance"] += 0.2 # PCU
                    # Add agents on carpool-layer
                    if (int(k[0]) == int(j["recSource"])-40000) and (int(k[1]) == int(j["recTarget"])-40000):
                        congestion[c][2]["resistance"] += 0.2 # PCU
                    # Add agents on bicycle-layer
                    if (int(k[0]) == int(j["recSource"])-20000) and (int(k[1]) == int(j["recTarget"])-20000):
                        congestion[c][2]["resistance"] += 0.2 # PCU


        # Update all links
        for row in links:

            resistance1 = [0, 0, 0, 0, 0, 0]
            resistance1[0] = row['resistance10']
            resistance1[1] = row['resistance11']
            resistance1[2] = row['resistance12']
            resistance1[3] = row['resistance13']
            resistance1[4] = row['resistance14']
            resistance1[5] = row['resistance15']

            resistance2 = [0, 0, 0, 0, 0, 0]
            resistance2[0] = row['resistance20']
            resistance2[1] = row['resistance21']
            resistance2[2] = row['resistance22']
            resistance2[3] = row['resistance23']
            resistance2[4] = row['resistance24']
            resistance2[5] = row['resistance25']

            speed = row['speed']
            try:
                nrOfAgents = congestion[c][2]["resistance"]
            except:
                nrOfAgents = 0

            capacity = row['capacity']
            try:
                dfBPRRow = dfBPR.loc[(dfBPR['PythonO'].astype('str') == row['source'][-4:]) & (dfBPR['PythonD'].astype('str') == row['target'][-4:])]
                capacity = float(dfBPRRow['Cfill']) # [veh/hour]
            except: # When moving between layers, no need for determining capacity
                pass


            # Look-up congestion on link
            for c, k in enumerate(congestion):
                if (k[0] == row['source'] and k[1] == row['target'] and ((i < updateResInt) or (i % updateResInt == 0))):
                    resistance1, resistance2, speed, nrOfAgents = calcResistance(congestion[c][2]["resistance"], row['distance'], capacity, row['defSpeed'], row['mode'], row['source'], row['target'], speed, i, timeStep, scalingSampleSize, futureCharRow, networkName)

            dictLink = {
                "source": row['source'],
                "target": row['target'],
                "resistance10": resistance1[0],
                "resistance11": resistance1[1],
                "resistance12": resistance1[2],
                "resistance13": resistance1[3],
                "resistance14": resistance1[4],
                "resistance15": resistance1[5],
                "resistance20": resistance2[0],
                "resistance21": resistance2[1],
                "resistance22": resistance2[2],
                "resistance23": resistance2[3],
                "resistance24": resistance2[4],
                "resistance25": resistance2[5],
                "capacity": row['capacity'],
                "distance": row['distance'],
                "speed": speed,
                "nrOfAgents": nrOfAgents,
                "defSpeed": row['defSpeed'],
                "mode": row['mode'],
            }
            dictLinks.append(dictLink)

        dictNodes = {
            "nodes": dictNodes,
            "links": dictLinks,
            "graph": {},
            "directed": 'False',
            "multigraph": 'False'
        }

        # Save all trips with details for current timestep
        with open('data/linksTimeStep' + str(i) + '.json', 'w', encoding='utf-8') as f:
            json.dump(dictNodes, f, ensure_ascii=False, indent=4)

        print("Timestep", i, "/", len(time), "completed")


def plotNetwork(time, maxCapacity):

    # time = np.arange(50)

    ############################ Plot network ################################

    # Load json networkx datafile
    with open('simpleNetwork.json','r') as infile:
        G = nx.json_graph.node_link_graph(json.load(infile))
    pos = nx.get_node_attributes(G, 'pos')
    print(G)

    # Simple network
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx_edges(G, pos, edgelist=[('0', '3'), ('0', '4'), ('0', '5'), ('0', '6'), ('0', '7'), ('3', '10'), ('10', '17'), ('17', '1'), ('4', '11'), ('11', '18'), ('18', '1'), ('5', '12'), ('12', '19'), ('19', '1'), ('6', '13'), ('13', '20'), ('20', '1'), ('7', '14'), ('14', '21'), ('21', '1')])
    # nx.draw_networkx_edges(G, pos, edge_color='0.75', connectionstyle="arc3,rad=0.2", edgelist=[('2', '3'), ('2', '4'), ('2', '5'), ('2', '6'), ('2', '7'), ('2', '8'), ('9', '10'), ('9', '11'), ('9', '12'), ('9', '13'), ('9', '14'), ('9', '15'), ('16', '17'), ('16', '18'), ('16', '19'), ('16', '20'), ('16', '21'), ('16', '22'), ('3', '2'), ('10', '9'), ('17', '16'), ('4', '2'), ('11', '9'), ('18', '16'), ('5', '2'), ('12', '9'), ('19', '16'), ('6', '2'), ('13', '9'), ('20', '16'), ('7', '2'), ('14', '9'), ('21', '16'), ('8', '2'), ('15', '9'), ('22', '16')])
    # nx.draw_networkx_edges(G, pos, style='dotted', edgelist=[('0', '8'), ('8', '15'), ('15', '22'), ('22', '1')])
    
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '3'): 'Car',('3', '10'): 'Car',('10', '17'): 'Car',('17', '1'): 'Car'}, font_color='red')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '4'): 'Carpool',('4', '11'): 'Carpool',('11', '18'): 'Carpool',('18', '1'): 'Carpool'}, font_color='orange')   
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '5'): 'Transit',('5', '12'): 'Transit',('12', '19'): 'Transit',('19', '1'): 'Transit'}, font_color='green')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '6'): 'Bicycle',('6', '13'): 'Bicycle',('13', '20'): 'Bicycle',('20', '1'): 'Bicycle'}, font_color='blue')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '7'): 'Walk',('7', '14'): 'Walk',('14', '21'): 'Walk',('21', '1'): 'Walk'}, font_color='purple')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '8'): 'Future',('8', '15'): 'Future',('15', '22'): 'Future',('22', '1'): 'Future'}, font_color='black')

    # plt.savefig('figures/simpleNetwork.png', dpi=300)
    plt.savefig('figures/DelftNetwork.png', dpi=300)



    # # With multimodal trip
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_labels(G, pos)
    # nx.draw_networkx_edges(G, pos, edgelist=[('0', '3'), ('0', '4'), ('0', '5'), ('0', '6'), ('0', '7'), ('3', '10'), ('10', '17'), ('17', '1'), ('4', '11'), ('11', '18'), ('18', '1'), ('5', '12'), ('12', '19'), ('19', '1'), ('6', '13'), ('13', '20'), ('20', '1'), ('7', '14'), ('14', '21'), ('21', '1')])
    # nx.draw_networkx_edges(G, pos, edge_color='0.75', connectionstyle="arc3,rad=0.2", edgelist=[('2', '3'), ('2', '4'), ('2', '5'), ('2', '6'), ('2', '7'), ('2', '8'), ('9', '10'), ('9', '11'), ('9', '12'), ('9', '13'), ('9', '14'), ('9', '15'), ('16', '17'), ('16', '18'), ('16', '19'), ('16', '20'), ('16', '21'), ('16', '22'), ('3', '2'), ('10', '9'), ('17', '16'), ('4', '2'), ('11', '9'), ('18', '16'), ('5', '2'), ('12', '9'), ('19', '16'), ('6', '2'), ('13', '9'), ('20', '16'), ('7', '2'), ('14', '9'), ('21', '16'), ('8', '2'), ('15', '9'), ('22', '16')])
    # nx.draw_networkx_edges(G, pos, style='dotted', edgelist=[('0', '8'), ('8', '15'), ('15', '22'), ('22', '1')])

    # nx.draw_networkx_edges(G, pos, edge_color='red', edgelist=[('0', '6'), ('3', '10'), ('12', '19'), ('21', '1')])
    # nx.draw_networkx_edges(G, pos, edge_color='red', connectionstyle="arc3,rad=0.2", edgelist=[('6', '2'), ('2', '3'), ('10', '9'), ('9', '12'), ('19', '16'), ('16', '21')])
    
    
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '3'): 'Car',('3', '10'): 'Car',('10', '17'): 'Car',('17', '1'): 'Car'}, font_color='red')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '4'): 'Carpool',('4', '11'): 'Carpool',('11', '18'): 'Carpool',('18', '1'): 'Carpool'}, font_color='orange')   
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '5'): 'Transit',('5', '12'): 'Transit',('12', '19'): 'Transit',('19', '1'): 'Transit'}, font_color='green')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '6'): 'Bicycle',('6', '13'): 'Bicycle',('13', '20'): 'Bicycle',('20', '1'): 'Bicycle'}, font_color='blue')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '7'): 'Walk',('7', '14'): 'Walk',('14', '21'): 'Walk',('21', '1'): 'Walk'}, font_color='purple')
    # nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={('0', '8'): 'Future',('8', '15'): 'Future',('15', '22'): 'Future',('22', '1'): 'Future'}, font_color='black')

    # plt.savefig('figures/simpleNetwork_route.png', dpi=300)

    ############################ Plot agents over time ################################

    modeLayers = [60000, 70000, 80000] # [30000, 40000, 50000, 60000, 70000, 80000]
    modes = ['Bicycle', 'Walk', 'Future'] # ['Car', 'Carpool', 'Transit', 'Bicycle', 'Walk', 'Future']
    speedModes = [14.621403427954009, 5, 10] # [34.668441679929636, 30.67680672911914, 29.074451916316153, 14.621403427954009, 5, 10] # From ODIN 2019

    for c, j in enumerate(modes):

        modeLayer = modeLayers[c]
        speedMode = speedModes[c]

        checkSpeed = 100

        for i in time:

            with open('linksTimeStep' + str(i) + '.json','r') as infile:
                G = nx.json_graph.node_link_graph(json.load(infile))
            pos = nx.get_node_attributes(G, 'pos')

            weights = [1 - ((G[u][v][0]['nrOfAgents'] * 66 / G[u][v][0]['distance'] / G[u][v][0]['capacity'])) for u,v in G.edges()]
            speedEdge = [G[u][v][0]['speed']/8 for u,v in G.edges()]

            # Only keep the widths of the relevant edges related to the mode being visualised
            cntr = 0
            for u,v in G.edges():
                if (int(u) < modeLayer) or (int(u) > modeLayer+9999):
                    speedEdge[cntr] = 0
                    weights[cntr] = 0
                if (int(v) < modeLayer) or (int(v) > modeLayer+9999):
                    speedEdge[cntr] = 0
                    weights[cntr] = 0
                cntr += 1

            for k in speedEdge:
                if (k*8 > 4) and (k*8 < checkSpeed):
                        checkSpeed = k*8

            # Plot the network
            fig = plt.figure()
            nx.draw_networkx_edges(G, pos, edge_color=weights, connectionstyle="arc3,rad=0.1", edge_cmap=plt.cm.RdYlGn, edge_vmin=0, edge_vmax=1, node_size=0, width=speedEdge)
            plt.savefig('figures/timestep/CapNetworkWithAgents' + j + '_' + str(i) + '.png', dpi=300)
            plt.close(fig)

            fig = plt.figure()
            nx.draw_networkx_edges(G, pos, edge_color=speedEdge, connectionstyle="arc3,rad=0.1", edge_cmap=plt.cm.RdYlGn, edge_vmin=0, edge_vmax=1, node_size=0, width=weights*7)
            plt.savefig('figures/timestep/SpeedNetworkWithAgents' + j + '_' + str(i) + '.png', dpi=300)
            plt.close(fig)


            print("Plotting stills timestep", i, "/", len(time), "completed")

            print('checkSpeed', checkSpeed)

        # Create gif from plots
        with iio.get_writer('figures/CapNetworkAgents' + j + '.gif', duration=0.05) as writer:
            for i in time:
                file = 'figures/timestep/CapNetworkWithAgents' + j + '_' + str(i) + '.png'
                image = iio.imread(file)
                writer.append_data(image)
                print("Plotting animation timestep", i, "/", len(time), "completed")

        with iio.get_writer('figures/SpeedNetworkAgents' + j + '.gif', duration=0.05) as writer:
            for i in time:
                file = 'figures/timestep/SpeedNetworkWithAgents' + j + '_' + str(i) + '.png'
                image = iio.imread(file)
                writer.append_data(image)
                print("Plotting animation timestep", i, "/", len(time), "completed")


def statsNetwork(futureCharRow, networkName, nameFuture, scalingSampleSize, scalingSampleSizeTrips, startTime, time, timeStep, runIteration, MNLbeta, MNLmu):

    # tripsAll = pd.read_json('trips.json')
    duration = []
    speedTrips = []
    totalDistanceTravelled = []
    distances = []

    # Calculate distance travelled per mode/link
    with open('data/simpleNetwork.json','r') as infile:
        G = nx.json_graph.node_link_graph(json.load(infile))
    edges = list(G.edges(data=True))
    edgeDf = []

    # Look-up link details
    for c, k in enumerate(edges):
        edgeDf.append([k[0], k[1], k[2]["distance"], k[2]["mode"]])
    
    df = pd.DataFrame(edgeDf, columns=['source', 'target', 'distance', 'mode'])
    tripsModeChoice = []
    tripsModeChoiceMixed = []
    agentsCluster = [0, 0, 0, 0, 0, 0]

    tripsData = pd.DataFrame()
    for i in time:
        tripsTimeStep = pd.read_pickle('data/tripsTimeStep' + str(i) + '.pkl')
        tripsData = pd.concat([tripsData, tripsTimeStep])

    # Add mode and distance to each trip segment
    tripsData['mode'] = 0
    tripsData['distance'] = 0
    for index in range(len(tripsData)):
        tripsData['mode'].iloc[index] = df['mode'].loc[(df['source'] == tripsData['recSource'].iloc[index]) & (df['target'] == tripsData['nextTarget'].iloc[index])].values[0]
        tripsData['distance'].iloc[index] = df['distance'].loc[(df['source'] == tripsData['recSource'].iloc[index]) & (df['target'] == tripsData['nextTarget'].iloc[index])].values[0]

    # Remove redundant columns
    tripsData = tripsData.drop(['pos_x', 'pos_y', 'positionLink'], axis=1)

    # Save results in archive to prevent overwriting of results
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y_%H:%M:%S_%f")[:-3]
    tripsData.to_csv("data/tripsDataAnalyse_" + networkName + '_' + nameFuture + '_' + date_time + ".csv")

    # tripsData = pd.read_pickle("tripsData.pkl")
    # tripsData = pd.read_csv("tripsDataAnalyse.csv", sep = ',', decimal = '.')

    # Extract modes and travel times
    uniqueTrips = tripsData['id'].unique()
    cntr = 0
    distancesTrips = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # car, carpool, transit, bicycle, walk, future, mixed, carMixed, carpoolMixed, transitMixed, bicyclesMidex, walkMixed, futureMixed

    for trip in uniqueTrips:

        tripDf = tripsData[tripsData['id'] == trip]
        occ = len(tripDf)

        # Average trip speed (distance / amount of occurrences)
        avTripSpeed = tripDf['speed'].mean()

        # Average trip duration (amount of occurrences * stepSize)
        avTripDuration = occ * timeStep

        # Trip distance
        dfTemp = tripDf.drop_duplicates('recSource')
        TripDistance = dfTemp['distance'].sum()

        # Modal split (count modes present in dataset)
        car = tripDf.loc[tripDf['mode'].str.contains("car",case=False)]
        carpool = tripDf.loc[tripDf['mode'].str.contains("pool",case=False)]
        transit = tripDf.loc[tripDf['mode'].str.contains("transit",case=False)]
        bicycle = tripDf.loc[tripDf['mode'].str.contains("bicycle",case=False)]
        walk = tripDf.loc[tripDf['mode'].str.contains("walk",case=False)]
        future = tripDf.loc[tripDf['mode'].str.contains("future",case=False)]

        # Modal split (veh*km)
        carD = dfTemp.loc[dfTemp['mode'].str.contains("car",case=False)]
        carpoolD = dfTemp.loc[dfTemp['mode'].str.contains("pool",case=False)]
        transitD = dfTemp.loc[dfTemp['mode'].str.contains("transit",case=False)]
        bicycleD = dfTemp.loc[dfTemp['mode'].str.contains("bicycle",case=False)]
        walkD = dfTemp.loc[dfTemp['mode'].str.contains("walk",case=False)]
        futureD = dfTemp.loc[dfTemp['mode'].str.contains("future",case=False)]

        carDistance = carD['distance'].sum()
        carpoolDistance = carpoolD['distance'].sum()
        transitDistance = transitD['distance'].sum()
        bicycleDistance = bicycleD['distance'].sum()
        walkDistance = walkD['distance'].sum()
        futureDistance = futureD['distance'].sum()

        binString = ''

        modesInTrip = 0

        if (not car.empty):
            binString = binString + '1'
            modesInTrip += 1
        else:
            binString = binString + '0'

        if (not carpool.empty):
            binString = binString + '1'
            modesInTrip += 1
        else:
            binString = binString + '0'

        if (not transit.empty):
            binString = binString + '1'
            modesInTrip += 1
        else:
            binString = binString + '0'

        if (not bicycle.empty):
            binString = binString + '1'
            modesInTrip += 1
        else:
            binString = binString + '0'

        if (not walk.empty):
            binString = binString + '1'
            modesInTrip += 1
        else:
            binString = binString + '0'

        if (not future.empty):
            binString = binString + '1'
            modesInTrip += 1
        else:
            binString = binString + '0'

        distances.append(TripDistance)

        if modesInTrip > 1: # Multimodal
            tripsModeChoice.append(7)
            distancesTrips[6] = distancesTrips[6] + TripDistance

            if not future.empty:
                tripsModeChoiceMixed.append(6)
                distancesTrips[12] = distancesTrips[12] + futureDistance

            if not walk.empty:
                tripsModeChoiceMixed.append(5)
                distancesTrips[11] = distancesTrips[11] + walkDistance

            if not bicycle.empty:
                tripsModeChoiceMixed.append(4)
                distancesTrips[10] = distancesTrips[10] + bicycleDistance

            if not transit.empty:
                tripsModeChoiceMixed.append(3)
                distancesTrips[9] = distancesTrips[9] + transitDistance

            if not carpool.empty:
                tripsModeChoiceMixed.append(2)
                distancesTrips[8] = distancesTrips[8] + carpoolDistance

            if not car.empty:
                tripsModeChoiceMixed.append(1)
                distancesTrips[7] = distancesTrips[7] + carDistance

        elif not future.empty:
            tripsModeChoice.append(6)
            distancesTrips[5] = distancesTrips[5] + futureDistance

        elif not walk.empty:
            tripsModeChoice.append(5)
            distancesTrips[4] = distancesTrips[4] + walkDistance

        elif not bicycle.empty:
            tripsModeChoice.append(4)
            distancesTrips[3] = distancesTrips[3] + bicycleDistance

        elif not transit.empty:
            tripsModeChoice.append(3)
            distancesTrips[2] = distancesTrips[2] + transitDistance

        elif not carpool.empty:
            tripsModeChoice.append(2)
            distancesTrips[1] = distancesTrips[1] + carpoolDistance

        elif not car.empty:
            tripsModeChoice.append(1)
            distancesTrips[0] = distancesTrips[0] + carDistance

        else: # Other, analysis failed
            tripsModeChoice.append(99)

        duration.append(avTripDuration)
        speedTrips.append(avTripSpeed)
        totalDistanceTravelled.append(TripDistance)

        # Add to cluster count
        for j in range(6):
            if (int(tripDf['cluster'].iloc[0]) == int(j)):
                agentsCluster[j] += 1 
        
        if cntr % 100 == 0:
            print("Calculation trip characteristics for trip ", cntr, "/", len(uniqueTrips), "completed")

        cntr += 1

    # Find total observed utility in network (note that utility1 and utility2 are summed up, this gives an indication, but utility2 component is dependent on the length of each route)
    utilityTimeStep = []

    for i in time:
        resistance = [0, 0, 0, 0, 0, 0]
        with open('data/linksTimeStep' + str(i) + '.json','r') as infile:
            G = nx.json_graph.node_link_graph(json.load(infile))
        pos = nx.get_node_attributes(G, 'pos')
        for j in range(6):
            resistanceArray1 = [G[u][v][0]['resistance1' + str(j)] for u,v in G.edges()]
            resistanceArray2 = [G[u][v][0]['resistance2' + str(j)] for u,v in G.edges()]
            resistance[j] = agentsCluster[j] * (np.sum(resistanceArray1) + np.sum(resistanceArray2)) / np.sum(agentsCluster)
        utilityTimeStep.append(np.sum(resistance))
    totalUtility = np.sum(utilityTimeStep)


    # Print and save all results
    tripsModeChoice = np.array(tripsModeChoice)
    tripsModeChoiceMixed = np.array(tripsModeChoiceMixed)
    distances = np.array(distances)
    tripDistRes = np.stack((tripsModeChoice, distances), axis=1)
    np.savetxt('data/tripDistRes.csv', tripDistRes, delimiter=',')

    print("Modal split: car:", (tripsModeChoice == 1).sum()/len(tripsModeChoice), 
        "carpool:", (tripsModeChoice == 2).sum()/len(tripsModeChoice), 
        "transit:", (tripsModeChoice == 3).sum()/len(tripsModeChoice), 
        "bicycle:", (tripsModeChoice == 4).sum()/len(tripsModeChoice), 
        "walk:", (tripsModeChoice == 5).sum()/len(tripsModeChoice), 
        "future:", (tripsModeChoice == 6).sum()/len(tripsModeChoice), 
        "mixed:", (tripsModeChoice == 7).sum()/len(tripsModeChoice),
        "Mixed car", (tripsModeChoiceMixed == 1).sum()/len(tripsModeChoiceMixed), 
        "Mixed carpool:", (tripsModeChoiceMixed == 2).sum()/len(tripsModeChoiceMixed), 
        "Mixed transit:", (tripsModeChoiceMixed == 3).sum()/len(tripsModeChoiceMixed), 
        "Mixed bicycle:", (tripsModeChoiceMixed == 4).sum()/len(tripsModeChoiceMixed), 
        "Mixed walk:", (tripsModeChoiceMixed == 5).sum()/len(tripsModeChoiceMixed), 
        "Mixed future:", (tripsModeChoiceMixed == 6).sum()/len(tripsModeChoiceMixed), 

        ) 
    print('Modal split sumcheck:', (1 == ((tripsModeChoice == 1).sum() + (tripsModeChoice == 2).sum() + (tripsModeChoice == 3).sum() + (tripsModeChoice == 4).sum() + (tripsModeChoice == 5).sum() + (tripsModeChoice == 6).sum() + (tripsModeChoice == 7).sum())/len(tripsModeChoice)))

    # Average, standard deviation travel time agents
    print("Average duration of trips:", np.average(duration), "[hr]")
    print("Standard deviation duration of trips:", np.std(duration), "[hr]")

    # Average, standard deviation speed agents
    print("Average speed of trips:", np.average(speedTrips), "[km/hr]")
    print("Standard deviation speed of trips:", np.std(speedTrips), "[km/hr]")

    # Total distance travelled
    print("Average distance travelled:", np.average(totalDistanceTravelled), "[km]")
    print("Total distance travelled:", np.sum(totalDistanceTravelled) * scalingSampleSizeTrips, "[km]")

    # Total observed utility
    print("Total observed utility", totalUtility, "[-]")

    # Save results in .csv
    header = ['dateTime', 'network', 'nameFuture', 'cost', 'speed', 'drivingTask', 'skills', 'weatherProtection', 'luggage', 'shared', 'availability', 'reservation', 'active', 'accessible', 'neutralToFuture', 'futureToNeutral', 'scalingSampleSize', 'seed', 
        'modalSplitCar', 'modalSplitCarpool', 'modalSplitTransit', 'modalSplitBicycle', 'modalSplitWalk', 'modalSplitFuture', 'modalSplitMixed', 
        'modalSplitMixedCar', 'modalSplitMixedCarpool', 'modalSplitMixedTransit', 'modalSplitMixedBicycle', 'modalSplitMixedWalk', 'modalSplitMixedFuture',
        'averageDuration', 'staDevDuration', 'medianDuration', 'averageSpeed', 'staDevSpeed', 'medianSpeed', 'totalDistanceTravelled', 'totalObservedUtility', 'nrOfTrips',
        'modalSplitCarKm', 'modalSplitCarpoolKm', 'modalSplitTransitKm', 'modalSplitBicycleKm', 'modalSplitWalkKm', 'modalSplitFutureKm', 'modalSplitMixedKm', 
        'modalSplitMixedCarKm', 'modalSplitMixedCarpoolKm', 'modalSplitMixedTransitKm', 'modalSplitMixedBicycleKm', 'modalSplitMixedWalkKm', 'modalSplitMixedFutureKm',]
    if runIteration == 0:
        with open('data/resultsSummary.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open('data/resultsSummary.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        now = datetime.now() # current date and time
        date_time = now.strftime("%m%d%Y_%H:%M:%S_%f")[:-3]

        data = [date_time, networkName, nameFuture, futureCharRow[0], futureCharRow[1], futureCharRow[2], futureCharRow[3], futureCharRow[4], futureCharRow[5], futureCharRow[6], futureCharRow[7], futureCharRow[8], futureCharRow[9], futureCharRow[10], futureCharRow[11], futureCharRow[12], scalingSampleSize, futureCharRow[13], 
            (tripsModeChoice == 1).sum()/len(tripsModeChoice), 
            (tripsModeChoice == 2).sum()/len(tripsModeChoice), 
            (tripsModeChoice == 3).sum()/len(tripsModeChoice), 
            (tripsModeChoice == 4).sum()/len(tripsModeChoice), 
            (tripsModeChoice == 5).sum()/len(tripsModeChoice), 
            (tripsModeChoice == 6).sum()/len(tripsModeChoice), 
            (tripsModeChoice == 7).sum()/len(tripsModeChoice), 
            (tripsModeChoiceMixed == 1).sum()/(tripsModeChoice == 7).sum(), 
            (tripsModeChoiceMixed == 2).sum()/(tripsModeChoice == 7).sum(), 
            (tripsModeChoiceMixed == 3).sum()/(tripsModeChoice == 7).sum(), 
            (tripsModeChoiceMixed == 4).sum()/(tripsModeChoice == 7).sum(), 
            (tripsModeChoiceMixed == 5).sum()/(tripsModeChoice == 7).sum(), 
            (tripsModeChoiceMixed == 6).sum()/(tripsModeChoice == 7).sum(), 
            np.average(duration), np.std(duration), np.median(duration),
            np.average(speedTrips), np.std(speedTrips), np.median(speedTrips), 
            np.sum(totalDistanceTravelled) * scalingSampleSizeTrips, totalUtility, len(tripsModeChoice),
            distancesTrips[0], distancesTrips[1], distancesTrips[2], distancesTrips[3], distancesTrips[4], distancesTrips[5], distancesTrips[6], distancesTrips[7], distancesTrips[8], distancesTrips[9], distancesTrips[10], distancesTrips[11], distancesTrips[12]]
        writer.writerow(data)
        f.close()

    print("Finished running one iteration of the simulation. Duration %s seconds" % round((timeOS.time() - startTime), 4))
    runIteration += 1
    return runIteration













