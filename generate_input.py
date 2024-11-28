import commands
import shutil
import os
import sys
import time
import linecache
#import numpy as np
import math

do_submit = True

delete_help_file = True # if True, delete all current jobs_*, out, err files

Ncell = 4     # linear size of supercell
N = Ncell*Ncell            # lattice size

#mus = [0.0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55, -0.6, -0.65, -0.7, -0.75, -0.8, -0.85, -0.9, -0.95, -1.0, -1.1, -1.2]#, -1.3, -1.4, -1.5]#, -0.08, -0.1, -0.12, -0.14, -0.16, -0.18, -0.2]#, -0.1, -0.12, -0.14, -0.16, -0.18, -0.2, -0.22, -0.24]#, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, \
#      -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8, -2.9, -3.0, \
#      -3.1, -3.2, -3.3, -3.4, -3.5]#, -3.6, -3.7, -3.8, -3.9, -4.0]
#mus = [0.0]
#mus = [-0.55, -0.6, -0.65, -0.7, -0.75, -0.8, -0.85, -0.9, -0.95, -1.0]
#mus = [0.0, -0.05, -0.1, -0.15, -0.2]
#mus = [-0.75, -0.8, -0.9, -1.0]
#mus = np.arange(1.2, 2.0, 0.2)
#mus = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]
mus = [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#mus = [1.0, 1.05, 1.1, 1.8, 2.0]
#mus = np.linspace(-1.0, 1.0, num=10)
#mus = [1.0, 1.0263157894736843, 1.0526315789473684, 1.0789473684210527, 1.1052631578947367, 1.131578947368421, 1.1578947368421053, 1.1842105263157894, 1.2105263157894737, 1.236842105263158, 1.263157894736842, 1.2894736842105263, 1.3157894736842106, 1.3421052631578947, 1.368421052631579, 1.3947368421052633, 1.4210526315789473, 1.4473684210526314, 1.4736842105263157, 1.5]
#mus = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2]
#mus = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2]
mus = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
mus = [0.22, 0.23, 0.25, 0.26, 0.28, 0.29, 0.31, 0.32, 0.34, 0.35, 0.37, 0.38, 0.4, 0.41, 0.43, 0.44, 0.46, 0.47, 0.49, 0.5, 0.52, 0.53, 0.55, 0.56, 0.58, 0.59, 0.61, 0.62, 0.64, 0.65, 0.67, 0.68, 0.7, 0.71, 0.73, 0.74, 0.76, 0.77, 0.79, 0.8]
#mus = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3]
#mus = [0.591, 0.592, 0.593, 0.594, 0.595, 0.596, 0.597, 0.598, 0.599]
#mus = [0.18]
#mus = [1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.7]
mus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
mus = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
mus = [1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2]
mus = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]
mus = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
mus = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9]
mus = [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
mus = [-0.39, -0.38, -0.37, -0.36, -0.35, -0.34, -0.33, -0.32, -0.31]
mus  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
mus = [-0.1]#0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26]
mus = [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
mus = [0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
mus = [0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
mus = [-2.0, -0.5, 0.0, 1.0, 2.0]
mus = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6]
#mus = [0.0, -0.4, -0.8, -1.2, -1.6, -2.0]

# IMPORTANT:
# parameters below are for TaS2 1T-1H geometry. Layer T is stars of David(SoD) and only center atom is considered in model.
# Layer H is perfect triangular lattice.
# original parameter values in Nature Communications | (2024) 15:1357
eh = -0.37
ets = [-0.1, -0.05, 0.0, 0.05, 0.1]    # onsite energy in layer T
ets = [0.0]
tH = 0.15     # in layer H
tT = 0.0   # NN hopping in SoDlayer T
Vints = [0.03, 0.06, 0.09]   # interlayer hopping
Vints = [0.03, 0.1, 0.3, 0.5, 1.0]
#Vints = [0.03]
Uh = 0.0   # interaction in layer H
Ut = 0.1   # interaction in layer T

localVs = [1.76, 1.77, 1.78, 1.79]#, 1.75, 1.8, 1.85, 1.9, 1.95]
localVs = [1.7, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9]
localVs = [0.5, 1.0, 1.5]#,2.5, 3.0, 3.5, 4.0]
localVs = [0.0]
#localVs = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
#localVs = [1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.7, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2.0]
#localVs = [1.745]

seeds = [1234567]#, 3234567]#, 3234567, 4234567, 5234567, 6234567, 7234567, 8234567, 9234567]

#! betas following: 
#          0.4, 0.45,  0.5, 0.55,  0.6,  0.7,  0.8,  0.9,   1,    1.2,   1.5,   1.6,   1.8,    2,    2.5,   3,    3.2,  3.5,  4,   4.2,   5,   6,   7,   8,   10,  15,    20,    25
Ls      = [40,   45,   50,   55,   30,   35,   40,   45,    40,    48,    30,    32,    36,    40,    50,   30,   32,   35,   40,   42,   50,  60,  70,  80,  100, 120,   160,   200]
dtaus   = [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02,  0.025, 0.025, 0.05, 0.05, 0.05,  0.05,  0.05,   0.1,  0.1,  0.1,  0.1,  0.1,  0.1, 0.1, 0.1, 0.1,  0.1, 0.125, 0.125, 0.125]
norths  = [10,   9,    10,   11,   10,   7,    10,   9,     10,    12,    10,    8,      8,    10,    10,   10,    8,    7,   10,    7,   10,  10,  10,  10,   10,  10,   10,    10]

#          1,    1.2,   1.5,   1.6,    2,   2.5,   3,    3.2,  3.5,  4,   4.2,   5,   6,   7,   8,   10,   12,    15,    20,    25,    30,    35,    40
Ls      = [40,    48,    30,    32,    40,   50,   30,   32,   35,   40,   42,   50,  60,  70,  80,  100,  96,    120,   160,   200,   240,   280,  320]
dtaus   = [0.025, 0.025, 0.05, 0.05, 0.05,  0.05, 0.1,  0.1,  0.1,  0.1,  0.1,  0.1, 0.1, 0.1, 0.1,  0.1, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
norths  = [10,    12,    10,    8,     10,   10,   10,    8,    7,   10,    7,   10,  10,  10,  10,   10,   8,    10,    10,    10,    10,    10,   10]

#! betas following: 
#          2,    3,   4,   5,   6,   7,   8,   10,  12.5,  20, 25
#Ls      = [20,  30,   40,  50,  60,  70,  80,  100, 100,   160]#,   200]
#dtaus   = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.125, 0.125]#, 0.125]
#norths  = [10,  10,   10,  10,  10,  10,  10,   10,  10,   10]#,    10]

#! betas following: 
#          6,   7,   8,   10,  12.5,  15
Ls      = [100]#, 180]#, 70,  80,  100, 100,   120]
dtaus   = [0.1]#25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.125, 0.125]
norths  = [10,  10,  10,   10,  10,   10]

#          10,  12.5,  15
#Ls      = [80, 100,   120]
#dtaus   = [0.125, 0.125, 0.125]
#norths  = [10,  10,   10]

#          5,   6,   7,   8,   10,    12.5,  15,    20,  25,   30,    40
#Ls      = [50,  60,  70,  80,  80,    100,   120,   160, 200]#,   240,  320]
#dtaus   = [0.1, 0.1, 0.1, 0.1, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
#norths  = [10,  10,  10,  10,   10,    10,   10,    10,     10,    10,   10]

#          15
#Ls      = [50]
#dtaus   = [0.05]
#norths  = [10]

#          30    40
#Ls      = [240,  320]
#dtaus   = [0.125, 0.125, 0.125, 0.125]
#norths  = [10,     10,    10,   10]

#          40
#Ls      = [40]
#dtaus   = [0.1]
#norths  = [10]

#          30
#Ls      = [240]
#dtaus   = [0.125]
#norths  = [10]

#! betas following for calculating entropy: 
#          0.01,   0.05,   0.1,   0.2,  0.3,  0.4, 0.45,  0.5, 0.55,  0.6,  0.7,  0.8,  0.9,   1,    1.2,   1.5,   1.6,   1.8, 2.2,  2.5,   2.8,  3.2,  3.5, 4.2,  5.5,  
#Ls      = [20,     20,     20,    20,   30,   40,   45,   50,   55,   30,   35,   40,   45,    40,    48,    30,    32,    36,  44,   50,   28,   32,   35,   42,  44]
#dtaus   = [0.0005, 0.0025, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02,  0.025, 0.025, 0.05, 0.05, 0.05, 0.05,  0.05, 0.1,  0.1, 0.1,  0.1, 0.125]
#norths  = [10,     10,     10,    10,   10,    10,   9,    10,   11,   10,   7,   10,    9,    10,    12,    10,    8,      8,  11,   10,    7,    8,    7,    7,  11]

#Ls      = [50,   56,   64,   70,   60,   40,  42,  50,  60,  70,  80,  100, 120, 160]#,   200]
#dtaus   = [0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.125]#, 0.125, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#norths  = [10,   8,    8,    10,   10,   10,  7,   10,  10,  10,  10,  10,  10,  10]#,    10]
#ntry    = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
#ntry2   = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
ntry    = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
ntry2   = [1,1,1,1,1,1,1,1,1]
ntry  = 0
ntry2 = 0

tdm = 1
HSF = -1      # -1 = random, 1 = from file 
nbin = 10      # only applies for non-MPI run
FTphy = 1    # if do FT for phy0

node = 1
walltime = "3:30:00"

def clean():

    for i in range(0,len(temp)):

        if os.path.exists("./T_" + str(temp[i])):
            shutil.rmtree("./T_" + str(temp[i]))

        cmd = "mkdir T_" + str(temp[i])
        os.system(cmd)

def prepare_input_file(filename, outputname, i, s, m):

    file = open(filename, "r")
    text = file.read()
    file.close()

    text = text.replace("OUTPUT"    , str(outputname))
    text = text.replace("GEOM"      , str(geomfile))

    if(Ls[i] >= 50):
        text = text.replace("NMEASval"    , str(10000))
    else:
        text = text.replace("NMEASval"    , str(100))

    if(Ls[i] >= 50):
        text = text.replace("NWARMval"   , str(2000))
    else:
        text = text.replace("NWARMval"   , str(20))

    text = text.replace("SEEDval"    , str(seeds[s]))
    text = text.replace("MUval"      , str(mus[m]))

    text = text.replace("Lval"    , str(Ls[i]))
    text = text.replace("DTAUval" , str(dtaus[i]))

    text = text.replace("NTRYval"   , str(ntry))
    text = text.replace("NTRY2val"  , str(ntry2))
    text = text.replace("TDMval"    , str(tdm))

    text = text.replace("NORTHval"  , str(norths[i]))

    text = text.replace("HSFval"    , str(HSF))
    text = text.replace("NBINval"   , str(nbin))

    text = text.replace("FTPHY" , str(FTphy))

    file = open(filename, "w")
    file.write(text)
    file.close()

def prepare_geom_file(geomfile, iVint, iet):

    file = open(geomfile, "r")
    text = file.read()
    file.close()

    text = text.replace("NCELL",  str(Ncell))
    text = text.replace("Uh" ,  str(Uh))
    text = text.replace("Ut" ,  str(Ut))
    text = text.replace("tT" ,  str(tT))
    text = text.replace("tH" ,  str(tH))
    text = text.replace("Vint" ,  str(Vints[iVint]))
    text = text.replace("eh" ,  str(eh))
    text = text.replace("et" ,  str(ets[iet]))

    file = open(geomfile, "w")
    file.write(text)
    file.close()

cmd = "rm jobs_*"
if(delete_help_file):
    os.system(cmd)

#cmd = "rm -r run_*"
#if(delete_help_file):
#    os.system(cmd)

#cmd = "rm out_*"
#if(False and delete_help_file):
#    os.system(cmd)

cmd = "rm error_*"
if(False and delete_help_file):
    os.system(cmd)

for iVint in range(len(Vints)):
    for iet in range(len(ets)):
        for iV in range(len(localVs)):
            localV = localVs[iV]
            #epV = ep-localV
            
            lattice  = "V"+str(Vints[iVint])+"_Uh"+str(Uh)+"_Ut"+str(Ut)+"_tH"+str(tH)+"_tT"+str(tT)+"_eh"+str(eh)+"_et"+str(ets[iet])+"_N"+str(N)
            geomfile = "geom_V"+str(Vints[iVint])+"_Uh"+str(Uh)+"_Ut"+str(Ut)+"_tH"+str(tH)+"_tT"+str(tT)+"_eh"+str(eh)+"_et"+str(ets[iet])+"_N"+str(N)
            print (lattice)

            cmd = "cp g_template "+ geomfile
            #print (cmd)
            os.system(cmd)

            iUp = 0
            prepare_geom_file(geomfile, iVint, iet)

            for m in range(0, len(mus)):
                for i in range(0, len(Ls)):

                    beta = Ls[i]*dtaus[i]
                    print ("L=", Ls[i], "dtau=", dtaus[i], "beta = ", beta, "mu = ", mus[m], "N = ", N)

                    dir = "./"+lattice+"_be" + str(beta) +"_mu" + str(mus[m])

                    if not os.path.exists(dir):
                        cmd = "mkdir " + dir
                        os.system(cmd)
                        cmd = "cp main.e " + dir
                        os.system(cmd)
                        cmd = "cp " + geomfile + " "+ dir
                        os.system(cmd)

                    for s in range(0,len(seeds)):
                        input_file_name    = dir + "/input_seed" + str(seeds[s])
                        data_file_name     = lattice + "_be" + str(beta)+ "_s" + str(seeds[s])+ "_mu" + str(mus[m])

                        cmd = "cp ./in_template " + input_file_name + ";"
                        os.system(cmd)
                        prepare_input_file(input_file_name, data_file_name, i, s, m)

                        batch_str = ""
                        if node==1:
                            batch_str = batch_str + "./main.e " + dir + "/input_seed"+ \
                                            str(seeds[s])+" > out_"+lattice+ "_be" + str(beta)+ "_s" + str(seeds[s])+ "_mu" + str(mus[m]) +"\n"
                        elif node>1:
                            batch_str = batch_str + "mpirun -np "+str(node)+" ./main.e " + dir + "/input_seed"+ \
                                            str(seeds[s]) +"\n"

                        file = open("batch_script.pbs", "r")
                        text = file.read()
                        file.close()

                        jobs_file_name = dir + "/jobs_"+lattice+"_be"+ str(beta)+ "_s"+str(seeds[s])+".pbs"
                    
                        text = text.replace("NODES"  , str(node))
                        text = text.replace("WALLTIME"  , str(walltime))
                        text = text.replace("MPI"   , str(node*8))
                        text = text.replace("BETA" , str(beta))
                        text = text.replace("SEED" , str(seeds[s]))
                        text = text.replace("JOBS" , str(batch_str))
                        text = text.replace("LATTICE" , lattice)

                        if(not batch_str == ""):
                            file = open(jobs_file_name, "w")
                            file.write(text)
                            file.close()
        
                        if(do_submit):
                            cmd = "qsub " + jobs_file_name
                            os.system(cmd)

