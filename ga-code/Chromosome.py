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
                            self._genes[i][1] += change
                            

                        elif(k <= 0.5 and k < 0.25):
                            self._genes[i][1] -= change
                        
                        elif(k > 0.5 and k <= 0.75):
                            self._genes[i][1] -= change
                            
                        elif(k > 0.75):
                            self._genes[i][1] += change
                            
                # Checking for intersection in slope of upper curve and lower curve anchor
                upper = np.array([self._genes[3], self._genes[4]])
                lower = np.array([self._genes[3], self._genes[2]])
                if(slopec.slope_checker(lower, upper)):
                    flag = True
                    print("Slope Check Passed")

                    self._genes[CHROMOSOME_SIZE - 2][0] = min_w + (max_w-min_w) * random.random()   # overhang
                    self._genes[CHROMOSOME_SIZE - 2][1] = min_d + ((max_d-min_d) * random.random()) # gap
                    #self._genes[CHROMOSOME_SIZE - 1][0] = -(20 + 35 * random.random())             # deflection
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
        """Running OpenFOAM CFD Simulation"""
        start=time.time()
        subprocess.call([cfd_call,str(self.I)]) 
        end=time.time()
        return (end-start)/60
        
        
    def get_cost(self):
        """Calculating Cost from the files written by OpenFOAM"""
        cfd_file = open(gcost% self.I, "r")  
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