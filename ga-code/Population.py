# class that create one set of generations
class Population:
    def __init__(self, size, gen_num=0, mode=" "):
        self._chromosomes = []
        gen_num           = str(gen_num)
        
        i = 0
        while i < size:
            self.add_chromosomes(Chromosome(i+1, int(gen_num), mode))
            i + = 1

    def add_chromosomes(self, chromosome):
        self._chromosomes.append(chromosome)

    def get_chromosomes(self):
        pop_chroms_2d_array = np.zeros(
            (len(self._chromosomes), CHROMOSOME_SIZE, 2), dtype=float)
        
        for i in range(len(self._chromosomes)):
            pop_chroms_2d_array[i] = self._chromosomes[i].get_genes()
        return pop_chroms_2d_array