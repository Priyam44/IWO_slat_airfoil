"""Author : Priyam Gupta"""


import random,os,math,copy,STLgen,aircordinates,subprocess
import GeneticAlgorithm, Chromosome, Population
import _print_population as print_pop
import matplotlib.pyplot as plt
import slopecheck as slopec
import bezierbuilder as bb
import pandas as pd
import numpy as np
import time

MAX_POPULATION_SIZE = 8 # maximum number of plants in a colony(or population)
#STANDARD Deviation
#SD for SHAPE
sigma_fin1 = 0.0005  # final standard deviation
sigma_ini1 = 10      # initial standard deviation
#SD for POSITION
sigma_fin2 = 0.0005  # final standard deviation
sigma_ini2 = 6       # initial standard deviation

Smin = 0             # min seeds produced
Smax = 3             # max seeds produced
n_mi = 3             # modulation index
iter_max = 30        # Maximum number of iterations to be done

# no of control points + [overhang,gap] + [deflection angle,fitness]
CHROMOSOME_SIZE = 8 + 2
change = 3

#design space
max_w = (15-1.85)*0.25
min_w = (3.4-1.85)*0.25
max_d = (3.85-3.5)*0.25
min_d = (3.85+4)*0.25

#max and min lines to be read from cfd file
MAX_Line = 10000
MIN_Line = 9500

#filepaths
cfdstl  ="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/cfd_openfoam/wingMotion_snappyHexMesh/constant/triSurface/air.stl" #cfd stl geometry
cu_file =r"/home/fmg/OpenFOAM/fmg-6/run/Airfoil/curves/"                          #file where curves have to be saved
cfd_call="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/cfd_openfoam/Allrun"               # cfd call path
gcost   ="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/simvalue/CP%i/forceCoeffs.dat"     #get cost file where simvalues are stored
delete  ="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/ga-code/delete"                    #delete bash script

def create_file(pathname, filename, openmode):
    filepath = os.path.join(pathname, filename)
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    file1 = open(pathname+filename, openmode)
    return file1



# main
# initialising population
s=time.time()
subprocess.call([delete])
print("Running for.......\nMAX_ITERATION:",iter_max)
print("POPULATION SIZE:",MAX_POPULATION_SIZE)
print("GENERATION#0-----------------------------------------------------------------------------------")
population = Population(MAX_POPULATION_SIZE, 0, "initialise")
#dplot1=plotter.DynamicUpdate()#plotting maximum fitness dynamically
GeneticAlgorithm.sort(population)
print_pop._print_population(population, 0)
iter = 1
sigma = [GeneticAlgorithm.std_deviation(0, iter_max,sigma_ini1,sigma_fin1)]
while iter < iter_max:
    print("**************************************EVOLUTION STARTED*********************************************************")
    sigma.append(GeneticAlgorithm.std_deviation(iter, iter_max,sigma_ini1,sigma_fin1))
    print("GENERATION#",iter,"-----------------------------------------------------------------------------------")
    print("sigma shape:",GeneticAlgorithm.std_deviation(iter,iter_max,sigma_ini1,sigma_fin1))
    print("sigma position:",GeneticAlgorithm.std_deviation(iter,iter_max,sigma_ini2,sigma_fin2))
    population,new_pop = GeneticAlgorithm.reproduce(population, iter)
    print("*************************************REPRODUCED*********************************************")
    if(population == False):
        iter+=1
        break
    print_pop._print_population(new_pop, iter)
    iter += 1

plt.figure()
plt.plot(np.arange(1, iter+1), sigma)
e=time.time()
print("TOTAL TIME TAKEN: ",((e-s)/60),"mins")
#plt.show()
