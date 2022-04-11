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