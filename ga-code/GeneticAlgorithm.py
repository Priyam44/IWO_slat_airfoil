# class that helps in evolving and mutating the genes of the chromosomes
class GeneticAlgorithm:

    @staticmethod
    def reproduce(pop, iter):
        new_pop     = copy.deepcopy(pop)
        worst_cost  = pop._chromosomes[-1].get_genes()[CHROMOSOME_SIZE - 1][1]
        best_cost   = pop._chromosomes[0].get_genes()[CHROMOSOME_SIZE - 1][1]
        sigma_iter1 = GeneticAlgorithm.std_deviation(iter, iter_max , sigma_ini1 , sigma_fin1) #for shape
        sigma_iter2 = GeneticAlgorithm.std_deviation(iter, iter_max , sigma_ini2 , sigma_fin2) #for position

        if(best_cost != worst_cost):
            seed_num = 0
            for i in range(MAX_POPULATION_SIZE): #limiting the number of individuals that can reproduce
                
                ratio = (pop._chromosomes[i]._genes[- 1][1] - worst_cost) / (best_cost - worst_cost)
                
                # number of seeds chromosome can produce on the basis of rank
                S = Smin + (Smax - Smin) * ratio
                for j in range(int(S)):
                    seed_num += 1
                    seed      = Chromosome(seed_num, iter)
                    flag      = False
                    while(not flag):
                        seed._genes[0][0] = pop._chromosomes[i].get_genes()[0][0]
                        seed._genes[0][1] = pop._chromosomes[i].get_genes()[0][1]
                        for k in range(1,CHROMOSOME_SIZE):
                            if (k<CHROMOSOME_SIZE-2):
                                seed._genes[k][0] = pop._chromosomes[i].get_genes()[k][0]
                                if (k <= 3 or k == 7):
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
        sindices = np.argsort(pop_chroms_2d_array[:, :, 1][:, -1], axis=0)
        print("sindices", sindices)
        sorted_chroms = Population(len(pop._chromosomes), 0, "zeros")
        for i in range(0, len(pop._chromosomes)):
            sorted_chroms._chromosomes[i]._genes = pop_chroms_2d_array[sindices[-(i+1)]]
        for i in range(0, len(pop._chromosomes)):
            pop._chromosomes[i] = sorted_chroms._chromosomes[i]