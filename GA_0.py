import random
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt


class Gene:
    """
    This is a class to represent individual(gene) in Genetic Algorithm
    """
    def __init__(self, **data):  # * -> tuple (), ** -> dictionary {}
        self.__dict__.update(data)
        self.size = len(data['data'])


class GA:
    """
    Genetic Algorithm
    """
    def __init__(self, parameter):
        self.parameter = parameter
        # parameter = [COPB, MUTPB, NGEN, popsize, low, up]
        low = self.parameter[4]
        up = self.parameter[5]
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)

        pop = []
        for i in range(self.parameter[3]):
            geneinfo = []
            for pos in range(len(low)):
                geneinfo.append(random.randint(self.bound[0][pos], self.bound[1][pos]))

            fitness = self.evaluate(geneinfo)
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness})
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)

    @staticmethod
    def evaluate(geneinfo):
        """
        fitness function
        :param geneinfo: [0]=x1, [1]=x2, [2]=x3, [3]=x4
        :return: y=fitness
        """
        x1 = geneinfo[0]
        x2 = geneinfo[1]
        x3 = geneinfo[2]
        x4 = geneinfo[3]
        y = x1 + x2 ** 2 + x3 ** 3 + x4 ** 4
        return y

    @staticmethod
    def selectBest(pop):
        """
        select the best individual from pop
        :param pop:
        :return:
        """
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]

    @staticmethod
    def selection(individuals, k):
        """
        select individuals by Roulette Wheel:
        individuals selected with a probability of its fitness
        :param individuals:
        :param k:
        :return:
        """
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)
        sum_fits = sum(ind['fitness'] for ind in individuals)

        chosen = []
        for i in range(k):
            point = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']
                if sum_ >= point:
                    chosen.append(ind)
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen

    @staticmethod
    def crossover(offspring):
        """
        crossover operation
        two points crossover
        :param offspring:
        :return:
        """
        dim = len(offspring[0]['Gene'].data)
        geneinfo1 = offspring[0]['Gene'].data
        geneinfo2 = offspring[1]['Gene'].data

        if dim == 1:
            pos1 = 1
            pos2 = 2
        else:
            pos1 = random.randrange(1, dim)
            pos2 = random.randrange(1, dim)

        newoff1 = Gene(data=[])
        newoff2 = Gene(data=[])
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(geneinfo2[i])
                temp1.append(geneinfo1[i])
            else:
                temp2.append(geneinfo1[i])
                temp1.append(geneinfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2

        return newoff1, newoff2

    @staticmethod
    def mutation(crossoff, bound):
        """
        mutation operation
        :param crossoff:
        :param bound:
        :return:
        """
        dim = len(crossoff.data)

        if dim == 1:
            pos = 0
        else:
            pos = random.randrange(0, dim)
        crossoff.data[pos] = random.randint(bound[0][pos], bound[1][pos])

        return crossoff

    def GA_main(self):
        """
        main frame work of GA
        :return:
        """
        popsize = self.parameter[3]
        print("Start of evolution")

        for g in range(NGEN):
            print("################### Generation {} ###################".format(g))
            selectpop = self.selection(self.pop, popsize)

            nextoff = []
            while len(nextoff) != popsize:
                offspring = [selectpop.pop() for _ in range(2)]
                if random.random() < COPB:
                    crossoff1, crossoff2 = self.crossover(offspring)
                    if random.random() < MUTPB:
                        muteoff1 = self.mutation(crossoff1, self.bound)
                        muteoff2 = self.mutation(crossoff2, self.bound)
                        fit_muteoff1 = self.evaluate(muteoff1.data)
                        fit_muteoff2 = self.evaluate(muteoff2.data)
                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                    else:
                        fit_crossoff1 = self.evaluate(crossoff1.data)
                        fit_crossoff2 = self.evaluate(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2})
                else:
                    nextoff.extend(offspring)

            self.pop = nextoff
            fits = [ind['fitness'] for ind in self.pop]
            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind

            print("Best individual found is {}, {}".format(self.bestindividual['Gene'].data,
                                                           self.bestindividual['fitness']))
            print("     Max fitness of current pop: {}".format(max(fits)))
        print("----- End of (successful) evolution ------")


# def plot_ga():
#     x = np.arange(0, 30, 1)
#     y = x + x ** 2 + x ** 3 + x ** 4
#     plt.plot(x, y)
#     plt.plot(, , "*")
#     plt.show()
#     # time.sleep(1)


if __name__ == "__main__":
    COPB, MUTPB, NGEN, popsize = 0.8, 0.1, 1000, 100
    low = [1, 1, 1, 1]
    up = [30, 30, 30, 30]
    parameter = [COPB, MUTPB, NGEN, popsize, low, up]
    run = GA(parameter)
    run.GA_main()
    # plot_ga()


