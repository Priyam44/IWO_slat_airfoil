"""two bezier curves"""
import random,os,math,copy,STLgen,aircordinates,subprocess
import numpy as np
import bezierbuilder as bb
import slopecheck as slopec
import matplotlib.pyplot as plt
import time
import pandas as pd
MAX_POPULATION_SIZE = 8 # maximum number of plants in a colony(or population)
#STANDARD Deviation
#SD for SHAPE
sigma_fin1 = 0.0005  # final standard deviation
sigma_ini1 = 10  # initial standard deviation
#SD for POSITION
sigma_fin2 = 0.0005  # final standard deviation
sigma_ini2 = 6  # initial standard deviation

Smin = 0  # min seeds produced
Smax = 3  # max seeds produced
n_mi = 3  # modulation index
iter_max = 30  # Maximum number of iterations to be done
# no of control points + [overhang,gap] + [deflection angle,fitness]
CHROMOSOME_SIZE = 8 + 2
change = 3
#design space
max_w=(15-1.85)*0.25
min_w=(3.4-1.85)*0.25
max_d=(3.85-3.5)*0.25
min_d=(3.85+4)*0.25
#max and min lines to be read from cfd file
MAX_Line=10000
MIN_Line=9500
#filepaths
cfdstl="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/cfd_openfoam/wingMotion_snappyHexMesh/constant/triSurface/air.stl"
cu_file=r"/home/fmg/OpenFOAM/fmg-6/run/Airfoil/curves/" #file where curves have to be saved
cfd_call="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/cfd_openfoam/Allrun" # cfd call path
gcost="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/simvalue/CP%i/forceCoeffs.dat" #get cost file where simvalues are stored
delete="/home/fmg/OpenFOAM/fmg-6/run/Airfoil/ga-code/delete"   #delete bash script
# class that generates chromosomes
def create_file(pathname, filename, openmode):
    filepath = os.path.join(pathname, filename)
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    file1 = open(pathname+filename, openmode)
    return file1


class Chromosome:
    def __init__(self, chrom_num=0, gen_number=0, mode=" ",):
        # creating initial gene
        self._genes = np.zeros((CHROMOSOME_SIZE, 2), dtype=float)
        self.I = (100*gen_number) + chrom_num
        if(mode == "initialise"):
            for i in range(CHROMOSOME_SIZE - 2):
                self._genes[i] = aircordinates.initial_cpoints[i]
            flag = False
            while (not flag):
                for i in range(1,CHROMOSOME_SIZE - 2):
                    if(i>3 and i<7):
                        # self._genes[i][0]=np.random.normal(initial_cpoints[i][0],0.5)
                        # self._genes[i][1]=np.random.normal(initial_cpoints[i][1],0.5)
                        k = random.random()

                        if(k <= 0.25):
                            #self._genes[i][0] += change
                            self._genes[i][1] += change
                             # print("++ ")gen_num: str

                        elif(k <= 0.5 and k < 0.25):
                            #self._genes[i][0] -= change
                            self._genes[i][1] -= change
                            #print("-- ")
                        elif(k > 0.5 and k <= 0.75):
                            #self._genes[i][0] += change
                            self._genes[i][1] -= change
                            #print("+- ")
                        elif(k > 0.75):
                            #self._genes[i][0] -= change
                            self._genes[i][1] += change
                            #print("-+ ")
                # checking for slope of upper curve and lower curve anchor
                # tangents
                upper = np.array([self._genes[3], self._genes[4]])
                lower = np.array([self._genes[3], self._genes[2]])
                if(slopec.slope_checker(lower, upper)):
                    flag = True
                    print("Slope Check Passed")

                    self._genes[CHROMOSOME_SIZE - 2][0] = min_w + (max_w-min_w) * random.random() # overhang
                    self._genes[CHROMOSOME_SIZE - 2][1] = min_d + ((max_d-min_d) * random.random()) # gap
                    #self._genes[CHROMOSOME_SIZE - 1][0] = -(20 + 35 * random.random()) # deflection
                    print("SEED#:",chrom_num)
                    curve=self.profile_gen()
                    Chromosome.genstl(curve,cfdstl)
                    time=self.cal_cost()
                    print("TIME TAKEN:",time," mins")
                    temp=self.get_cost()
                    if(temp==False):
                        flag=False
                    else:
                        self._genes[-1][1] = temp
                        print("acceptable individual")
                        print("\nCOST: ",self._genes[-1][1])
                        print("\nCurve Coordinates\n", curve)
                        break

                else:
                    print("Slope Check NOT Passed")
    def cal_cost(self):
        start=time.time()
        subprocess.call([cfd_call,str(self.I)]) #ADD ALLRUN FILENAME HERE
        end=time.time()
        return (end-start)/60
        
        #for debugging
        # return random.randint(1,6000)
    def get_cost(self):
        cfd_file = open(gcost% self.I, "r")  #ADD FORCECOEFF FILE PATH HERE
        cost=0
        list=cfd_file.readlines()
        #actual code
        if(len(list)>=MAX_Line):
            for i in range(MIN_Line,MAX_Line):
            	cost+=float(list[i].split()[3])

            cost=cost/(MAX_Line-MIN_Line)
            cfd_file.close()

            return cost
        else:
            print("SIMULATION INCOMPLETE........REGENERATING INDIVIDUAL")
            return False

        #for debugging

        # return random.randint(1,6000)
    def get_genes(self):
        return self._genes
    def profile_gen(self):
        cpoints = self.divide_cpoints()
        curve1 = bb.BezierBuilder(cpoints[0])
        curve2 = bb.BezierBuilder(cpoints[1])
        curve = np.array(curve1.get_coordinates())
        curve = np.append(curve, curve2.get_coordinates(), 0)
        curve = np.append(curve.transpose(), np.zeros((1,curve.shape[0])), 0).transpose() #adding z coordinates =0

        curve = self.transform_slat(curve,self._genes[-2][1], self._genes[-2][0],
                               self._genes[-1][0], self._genes[0], self._genes[3],self._genes)  # transforming slat
        # creating stl file of slat
        curve = curve/1000

        return curve
    @staticmethod
    def genstl(curve,filename):
        STLgen.STL_Gen(curve.transpose()[0],curve.transpose()[1],'new_slat.stl')
        #BELOW LINE TAKES THREE ARGUMENTS FIRST:SLAT FILE, SECOND: AIRFOIL FILE , THIRD: COMBINED FILE PATH along with name
        STLgen.combine('new_slat.stl','ClarkY_airfoil_surf.STL',filename)
        print("profile stl generated and saved")
    @staticmethod
    def transform_slat(target, gap, overhang, deflection, lead, trail,cpoints):
        """# making slat horizontal
        slope = math.degrees(
            math.atan(abs((lead[1] - trail[1]) / (lead[0] - trail[0]))))
        print(target)
        target = STLgen.rotate(target, slope, [0, 0, 1])"""

        """# getting tail to origin
        ysub = target[np.argmax(target, axis=0)[0]][1]
        target = target.transpose()
        target[0] = target[0] - target.max(1)[0]
        target[1] = target[1] - ysub"""
        #getting head to origin
        target= target.transpose()
        print (cpoints[0][0])
        target[0]-=  6.9961086 - cpoints[0][0]
        target[1]-=  -5.54978974 + cpoints[0][1]
        # providing deflection
        target = STLgen.rotate(target.transpose(), deflection, [0, 0, 1])
        # providing gap and overhang
        target = target.transpose()
        height=gap
        #height = (abs(gap**2 - overhang**2))**0.5
        target[0] = target[0] - overhang
        target[1] = target[1] + height
        """# providing angle of attack
        target = STLgen.rotate(target.transpose(), 13)"""
        return target.transpose()
    def divide_cpoints(self):
        cpoints1 = np.empty((0, 2))
        cpoints2 = np.empty((0, 2))
        for i in range(CHROMOSOME_SIZE):
            if(i <= 3):
                cpoints1 = np.append(cpoints1, [self._genes[i]], axis=0)
            if(i >= 3 and i <= 7):
                cpoints2 = np.append(cpoints2, [self._genes[i]], axis=0)
        cpoints2 = np.append(cpoints2, [self._genes[0]], axis=0)
        cpoints = np.array([cpoints1, cpoints2])
        return cpoints
    def __str__(self):
        return self._genes.__str__()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# class that create one set of generations
class Population:
    def __init__(self, size, gen_num=0, mode=" "):
        self._chromosomes = []
        i = 0
        gen_num = str(gen_num)
        while i < size:
            self.add_chromosomes(Chromosome(i+1, int(gen_num), mode))
            i += 1

    def add_chromosomes(self, chromosome):
        self._chromosomes.append(chromosome)

    def get_chromosomes(self):
        pop_chroms_2d_array = np.zeros(
            (len(self._chromosomes), CHROMOSOME_SIZE, 2), dtype=float)
        #pop_chroms_2d_array = np.around(pop_chroms_2d_array, 6)
        for i in range(len(self._chromosomes)):
            pop_chroms_2d_array[i] = self._chromosomes[i].get_genes()
        return pop_chroms_2d_array
# class that helps in evolving and mutating the genes of the chromosomes


class GeneticAlgorithm:
    @staticmethod
    def reproduce(pop, iter):
        new_pop = copy.deepcopy(pop)
        worst_cost = pop._chromosomes[-1].get_genes()[CHROMOSOME_SIZE - 1][1]
        best_cost = pop._chromosomes[0].get_genes()[CHROMOSOME_SIZE - 1][1]
        sigma_iter1 = GeneticAlgorithm.std_deviation(iter, iter_max , sigma_ini1 , sigma_fin1) #for shape
        sigma_iter2 = GeneticAlgorithm.std_deviation(iter, iter_max , sigma_ini2 , sigma_fin2) #for position
        if(best_cost != worst_cost):
            seed_num = 0
            for i in range(MAX_POPULATION_SIZE): #limiting the number of individuals that can reproduce
                ratio = (pop._chromosomes[i]._genes[- 1][1] - worst_cost) / (best_cost - worst_cost)
                # number of seeds chromosome can produce on the basis of rank
                S = Smin + (Smax - Smin) * ratio
                #print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS",int(S))
                for j in range(int(S)):
                    seed_num += 1
                    seed = Chromosome(seed_num, iter)
                    flag=False
                    while(not flag):
                        seed._genes[0][0] = pop._chromosomes[i].get_genes()[0][0]
                        seed._genes[0][1] = pop._chromosomes[i].get_genes()[0][1]
                        for k in range(1,CHROMOSOME_SIZE):
                            if (k<CHROMOSOME_SIZE-2):
                                seed._genes[k][0] = pop._chromosomes[i].get_genes()[k][0]
                                if (k<=3 or k==7):
                                    seed._genes[k][1] = pop._chromosomes[i].get_genes()[k][1]
                                else:
                                    seed._genes[k][1] = np.random.normal(
                                        pop._chromosomes[i].get_genes()[k][1], sigma_iter1)
                            elif(k>=CHROMOSOME_SIZE-2):
                                seed._genes[k][0] = np.random.normal(
                                    pop._chromosomes[i].get_genes()[k][0], sigma_iter2)
                                if(k!=CHROMOSOME_SIZE-1):
                                    seed._genes[k][1] = np.random.normal(
                                        pop._chromosomes[i].get_genes()[k][1], sigma_iter2)
                        # checking for slope of upper curve and lower curve anchor
                        # tangents
                        upper = np.array([seed._genes[3], seed._genes[4]])
                        lower = np.array([seed._genes[3], seed._genes[2]])
                        if(slopec.slope_checker(lower, upper)):
                            flag=True
                            print("SEED# ",seed_num)
                            print("Slope Check Passed")
                            curve=seed.profile_gen()
                            Chromosome.genstl(curve,cfdstl)
                            time = seed.cal_cost()
                            print("\nTIME TAKEN:",time," mins")
                            temp=seed.get_cost()
                            if(temp==False):
                                flag= False
                            else:
                                seed._genes[-1][1] = temp
                                #print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS",temp)
                                print("ACCEPTABLE INDIVIDUAL")
                                print("\nCOST: ",seed._genes[-1][1])
                                print("\nCurve Coordinates\n", curve)
                                new_pop.add_chromosomes(seed)
                        else:
                            print("Slope Check NOT PASSED")
            GeneticAlgorithm.sort(new_pop)
            for i in range(MAX_POPULATION_SIZE):
                pop._chromosomes[i] = new_pop._chromosomes[i]
        else:
            print("best and worst cost equal can`t reproduce")
            return False,False
        return pop,new_pop

    @staticmethod
    def std_deviation(iter, iter_max,sigma_ini,sigma_fin):
        sigma_iter = (((iter_max - iter)**n_mi) / iter_max **
                      n_mi) * (sigma_ini - sigma_fin) + sigma_fin
        return sigma_iter

    @staticmethod
    def sort(pop):
        pop_chroms_2d_array = pop.get_chromosomes()
        #print("chroms",pop.get_chromosomes())
        sindices = np.argsort(pop_chroms_2d_array[:, :, 1][:, -1], axis=0)
        print("sindices", sindices)
        sorted_chroms = Population(len(pop._chromosomes), 0, "zeros")
        for i in range(0, len(pop._chromosomes)):
            sorted_chroms._chromosomes[i]._genes = pop_chroms_2d_array[sindices[-(i+1)]]
        for i in range(0, len(pop._chromosomes)):
            pop._chromosomes[i] = sorted_chroms._chromosomes[i]

#------------------------------------------------------------------------------------------------------------------------------------#-



#------------------------------------------------------------------------------------------------------------------------------------#-


def _print_population(new_pop, gen_number):
    chroms = new_pop.get_chromosomes()
    print("\n---------------------------------------------------------")
    print("PRINTING AFTER SORTING THE POPULATION")
    print("Generation#", gen_number, "|Fittest chromosome fitness:",chroms[0][-1][1])
    print("-----------------------------------------------------------")
    #dplot1.update(gen_number,chroms[0][-1][1])

    i = 0
    for i in range(MAX_POPULATION_SIZE):
        print("PLANT#", i + 1, " ", "||Fitness:",chroms[i][-1][1], "\n")
        print("CONTROL POINTS\n",chroms[i])
        #handle.write("PLANT NO%i")
        print("--------------------------------------------------------------")
    i=0
    for i in range(len(new_pop._chromosomes)):
        I = (100*gen_number)+i+1
        curve_file = create_file(cu_file,"curve_%04i.dat" % I, 'w+')  # saving curve coordinates
        for j in range(CHROMOSOME_SIZE):
            curve_file.write(str(new_pop._chromosomes[i]._genes[j])+'\n')
        curve_file.close()
        plt.figure()
        curve=new_pop._chromosomes[i].profile_gen()
        Chromosome.genstl(curve,cu_file+"cstl_%04i.stl"%I)
        plt.plot(curve.transpose()[0],curve.transpose()[1])
        df = pd.read_csv('aircord.csv')
        x = df['X']
        y = df['Y']
        plt.plot(x/1000,y/1000,'.')
        plt.axes().set_aspect('equal')
        plt.xlim(-0.04,0.05)
        plt.ylim(-0.05,0.07)
        plt.plot(x,y,'r','.')
        #plt.plot(aircordinates.transpose()[0],aircordinates.transpose()[1],'r')
        plt.savefig(cu_file+"fig_%04i"%I)
    print("Curve File Saved")
#-------------------------------------------------------------------------





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
#handle=open(r"COST.dat",'w+')
#handle.write("Generation# 0 \n")
_print_population(population, 0)
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
    _print_population(new_pop, iter)
    iter += 1
plt.figure()
plt.plot(np.arange(1, iter+1), sigma)
e=time.time()
print("TOTAL TIME TAKEN: ",((e-s)/60),"mins")
#plt.show()
