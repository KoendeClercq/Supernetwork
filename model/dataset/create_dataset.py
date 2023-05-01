#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 5 14:45:13 2022

@author: gkdeclercq   

This script can be used to create a dataset ready for a supernetwork. It uses OViN, an calibrated OmniTRANS model and input from TNO 
to come up with a dataset for Delft with an OD-matrix including personal characteristics per trip.

"""

import time as timeOS
start_time = timeOS.time()

import csv
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

############################ PARAMETERS #################################

modes = ['car', 'carpool', 'transit', 'bicycle', 'walk']
postcodes = [2611, 2612, 2613, 2614, 2616, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629]

############################ FUNCTIONS ##################################

# Load ODiN 2019 dataset

df = pd.read_csv("ODIN2019/ODiN2019_Databestand_v2.csv", encoding = 'unicode_escape', sep=';')

# Calculate average speed per mode

df["SnelheidOP"] = (df["AfstandOP"] / 10) / (df["ReisduurOP"] / 60) # [km/h]
speed = [0, 0, 0, 0, 0]
speed[0] = df.loc[df["KHvm"] == '1']["SnelheidOP"].mean(axis=0)
speed[1] = df.loc[df["KHvm"] == '2']["SnelheidOP"].mean(axis=0)
speed[2] = df.loc[(df["KHvm"] == '3') | (df["KHvm"] == '4')]["SnelheidOP"].mean(axis=0)
speed[3] = df.loc[df["KHvm"] == '5']["SnelheidOP"].mean(axis=0)
speed[4] = min(df.loc[df["KHvm"] == '6']["SnelheidOP"].mean(axis=0), 5) # ODiN reports 13.3 km/h for walking, 5 km/h is more realistic

# print(speed)

# Remove all entries with KHvm 'other/nan'

df = df[df["KHvm"] != '7']
df = df[df["KHvm"] != '#NULL!']

df["CHOICE"] = df["KHvm"]
df.loc[df["KHvm"] == '4', "CHOICE"] = '3'
df.loc[df["KHvm"] == '5', "CHOICE"] = '4'
df.loc[df["KHvm"] == '6', "CHOICE"] = '5'

# Remove all weekendtrips
df = df[df["Weekdag"] != '1']
df = df[df["Weekdag"] != '7']


# Check modal split of ODIN2019 in Delft postcodes only
postcodesTemp = [2611.0, 2612.0, 2613.0, 2614.0, 2616.0, 2622.0, 2623.0, 2624.0, 2625.0, 2626.0, 2627.0, 2628.0, 2629.0, 2636.0, 2635.0, 2286.0, 2288.0, 2289.0, 2497.0, 2498.0, 2645.0, 3046.0]
for PC in postcodesTemp:
    df = df[df["VertPC"].isin(postcodesTemp)]
    df = df[df["AankPC"].isin(postcodesTemp)]

dfCar = df[df["CHOICE"] == '1']
sumCar = dfCar["FactorV"].sum()
dfCarpool = df[df["CHOICE"] == '2']
sumCarpool = dfCarpool["FactorV"].sum()
dfTransit = df[df["CHOICE"] == '3']
sumTransit = dfTransit["FactorV"].sum()
dfBicycle = df[df["CHOICE"] == '4']
sumBicycle = dfBicycle["FactorV"].sum()
dfWalk = df[df["CHOICE"] == '5']
sumWalk = dfWalk["FactorV"].sum()
modalSplit = [dfCar["FactorV"].sum(), dfCarpool["FactorV"].sum(), dfTransit["FactorV"].sum(), dfBicycle["FactorV"].sum(), dfWalk["FactorV"].sum()]

# Load OD-matrix from OmniTRANS model

zones = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 2611, 2612, 2613, 2614, 2616, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 3000]

ODMatrix = np.array([[0,   320, 120, 3,   6,   51,  14,  2428,    3872,    1152,    206, 3003,    6,   9,   92,  53,  1,   6,   176, 4,   78],
    [1380,    0,   62,  1,   0,   1,   0,   56,  27,  98,  52,  16,  1,   1,   8,   13,  0,   0,   4,   0,   5],
    [589, 70,  0,   69,  1,   6,   1,   252, 121, 437, 235, 11,  8,   4,   35,  110, 1,   10,  18,  0,   23],
    [9,   1,   44,  0,   2,   2,   0,   29,  6,   55,  88,  1,   45,  4,   68,  413, 3,   15,  7,   1,   7],
    [5,   0,   0,   1,   0,   1744,    29,  72,  38,  62,  8,   4,   178, 573, 244, 99,  50,  667, 1037,    2823,    15],
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [8,   0,   0,   0,   18,  628, 0,   38,  20,  9,   2,   47,  0,   1,   12,  5,   0,   6,   97,  7,   2],
    [2647,    14,  56,  10,  83,  753, 73,  0,   4662,    1871,    821, 422, 57,  106, 1047,    519, 9,   475, 2573,    52,  1147],
    [3636,    6,   23,  2,   38,  345, 33,  4021,    0,   1905,    341, 2594,    5,   19,  88,  43,  2,   84,  1178,    24,  96],
    [1428,    28,  110, 22,  81,  349, 19,  2131,    2515,    0,   4405,    75,  124, 199, 1970,    975, 17,  541, 1195,    51,  1309],
    [219, 13,  51,  30,  9,   54,  3,   805, 386, 3806,    0,   35,  173, 31,  304, 1571,    11,  68,  184, 6,   201],
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [9,   1,   3,   23,  297, 70,  1,   82,  8,   157, 253, 1,   0,   2538,   535, 3225,    1640,    513, 240, 187, 16],
    [13,  0,   1,   2,   849, 200, 3,   135, 28,  223, 40,  3,   2243,    0,   1075,    435, 633, 1462,    685, 534, 32],
    [124, 2,   10,  30,  351, 319, 29,  1305,    126, 2160,    384, 13,  460, 1046,    0,   4200,    92,  2329,    1093,    221, 310],
    [77,  5,   33,  191, 153, 139, 13,  697, 67,  1154,    2158,    8,   3001,    458, 4540,    0,   189, 1021,    478, 96,  165],
    [1,   0,   0,   2,   96,  23,  0,   15,  3,   25,  18,  0,   1883,    822, 122, 233, 0,   165, 78,  60,  4],
    [6,   0,   2,   6,   804, 731, 12,  493, 101, 495, 72,  8,   370, 1190,    1952,    791, 104, 0,   2506,    505, 117],
    [115, 1,   2,   2,   721, 6531,    110, 1544,    819, 630, 112, 74,  100, 322, 527, 213, 28,  1439,    0,   453, 149],
    [3,   0,   0,   0,   2689,    634, 11,  43,  23,  37,  5,   2,   107, 344, 146, 59,  30,  400, 622, 0,   6],
    [105, 2,   6,   3,   21,  90,  5,   1419,    138, 1423,    254, 17,  14,  31,  309, 153, 3,   140, 309, 9,   0]])


# Load population per zone in Delft from TNO
df_population = pd.read_table('population_generated_iteration_1_20220611T003439.txt')
df_population = df_population.loc[df_population['location_id'].isin(postcodes)]
totalPopulation = len(df_population)
df_population.to_csv('population.csv')

# Scale OD-matrix from OmniTRANS model with population per zone in Delft from TNO

# Scale ODiN dataset to match OD-matrix

# Load population

# 1. Get trip from odin2019
# 2. Multiply with factorV (verplaatsing)
# 3. Multiply with fraction of (population in postcode (origin)) / (total population NL)
# 4. Divide by number of weekdays in 2019
# 5. Add cluster number + create copies of each entry to match the 'FactorPostcode'
# 6. Randomly distribute trips over destinations based on OD-matrix proportions (assuming averages of personal attributes, choice is still captured in included dataset)

nrOfWeekdays = 261 # in 2019

for i in range(len(postcodes)):
    df['Factor' + str(postcodes[i])] = df["FactorV"] * len(df_population.loc[df_population['location_id'] == postcodes[i]]) / totalPopulation / nrOfWeekdays # Excl. df["FactorP"]?

# Add clusters to dataset based on DCM

# Two out of 6 clusters were based on trip purpose ((1) home (cl #2), (2) business + work (cl #1)). 
# Three other clusters had a trip purpose of ‘other’, where one cluster 
# only contained trips with people that do not own a car (cl #0) and the other two 
# clusters contained trips with people that own a car. These two final 
# clusters were differentiated by the information that people are (cl #5) or are not 
# the main car user (cl #4). The sixth cluster had a high car ownership and precipitation.
# Precipitation data was not available in the dataset for the supernetwork (cluster #3), so
# this cluster was left out (1172 out of 60035 samples).

df.loc[((df["Doel"] != '1') & (df["Doel"] != '2') & (df["Doel"] != '3') & (df["Doel"] != '4')) & ((df['OPBezitVm'] != 1) & (df['OPBezitVm'] != 2) & (df['OPBezitVm'] != 3)) & (df['AutoEig'] != 1), "cluster"] = 4 # Purpose: other, car ownership, not main car user
df.loc[((df["Doel"] != '1') & (df["Doel"] != '2') & (df["Doel"] != '3') & (df["Doel"] != '4')) & ((df['OPBezitVm'] != 1) & (df['OPBezitVm'] != 2) & (df['OPBezitVm'] != 3)) & (df['AutoEig'] == 1), "cluster"] = 5 # Purpose: other, car ownership, main car user
df.loc[((df["Doel"] != '1') & (df["Doel"] != '2') & (df["Doel"] != '3') & (df["Doel"] != '4')) & ((df['OPBezitVm'] == 1) | (df['OPBezitVm'] == 2) | (df['OPBezitVm'] == 3)), "cluster"] = 0 # Purpose: other, no car ownership
df.loc[df["Doel"] == '1', "cluster"] = 2 # Home
df.loc[(df["Doel"] == '2') | (df["Doel"] == '3') | (df["Doel"] == '4'), "cluster"] = 1 # Work & business

# Determine nrOfTrips per OD-pair per cluster using the type of people living in each zone
dataset = [['O', 'D', 'cluster', 'nrOfTrips']]
trips = []
for c1, i in enumerate(zones): # Origin
    # Find cluster sizes for type of people living in zone
    cl = [0, 0, 0, 0, 0, 0]
    for k in range(6):
        if i in postcodes:
            cl[k] = df['FactorV'].loc[(df['cluster'] == k + 1) & (df['VertPC'] == i)].sum() / df['FactorV'].loc[df['VertPC'] == i].sum()
        else:
            # If not in the 13 postal codes, average of the NL is assumed
            cl[k] = df['FactorV'].loc[df['cluster'] == k + 1].sum() / df['FactorV'].sum()

    # print(cl)

    for c2, j in enumerate(zones): # Destination
        # Assumed to be the same cluster distribution as with origin for all destinations (e.g., 2624 cluster distribution is different for 2611)
        for k in range(6):
            if i in postcodes:
                idx = postcodes.index(i)
                nrOfTrips = df['Factor' + str(postcodes[idx])].sum() * ODMatrix[c2][c1] / ODMatrix.sum(axis=1)[c1] * cl[k]
            else: # Ratio of OD-matrix used to determined nrOfTrips
                nrOfTrips = df['Factor' + str(postcodes[0])].sum() * ODMatrix[c2][c1] / ODMatrix.sum(axis=1)[c1] * cl[k] * ODMatrix.sum(axis=1)[c1] / ODMatrix.sum(axis=1)[0]

            # print(nrOfTrips)

            # Write in dataset
            try:
                dataset = np.append(dataset, [[i, j, k + 1, int(nrOfTrips)]], axis=0)

                for m in range(int(nrOfTrips)):
                    dictTrips = {
                        "origin": i,
                        "destination": j,
                        "cluster": k,
                    }
                    trips.append(dictTrips)
            except: # Skip inf and nan values (0 in original OD-matrix)
                pass

# Create synthetic trips
# trips = []
# for m in range(996):
#     dictTrips = {
#         "origin": 1001,
#         "destination": 1002,
#         "cluster": m % 6,
#     }
#     trips.append(dictTrips)

with open('trips.json', 'w', encoding='utf-8') as f:
    json.dump(trips, f, ensure_ascii=False, indent=4)

with open("dataset.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(dataset)

end_time = timeOS.time()
print("It took", int(end_time - start_time), "seconds to create the dataset.")