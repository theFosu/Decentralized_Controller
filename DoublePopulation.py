from __future__ import print_function

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues


class CompleteExtinctionException(Exception):
    pass


class DoublePopulation(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
        Difference between this and original NEAT implementation: two types or genomes can be assessed at the same time
    """

    def __init__(self, config1, config2, initial_state=None):
        self.reporters1 = ReporterSet()
        self.reporters2 = ReporterSet()
        self.config1 = config1
        self.config2 = config2

        stagnation1 = config1.stagnation_type(config1.stagnation_config, self.reporters1)
        self.reproduction1 = config1.reproduction_type(config1.reproduction_config,
                                                       self.reporters1,
                                                       stagnation1)
        if config1.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config1.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config1.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config1.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config1.fitness_criterion))

        stagnation2 = config2.stagnation_type(config2.stagnation_config, self.reporters2)
        self.reproduction2 = config2.reproduction_type(config2.reproduction_config,
                                                       self.reporters2,
                                                       stagnation2)
        if config2.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config2.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config2.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config2.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config2.fitness_criterion))

        if initial_state is None:
            self.generation = 0
            # Create a population from scratch, then partition into species.
            self.population1 = self.reproduction1.create_new(config1.genome_type,
                                                             config1.genome_config,
                                                             config1.pop_size)
            self.species1 = config1.species_set_type(config1.species_set_config, self.reporters1)
            self.species1.speciate(config1, self.population1, self.generation)

            self.population2 = self.reproduction1.create_new(config2.genome_type,
                                                             config2.genome_config,
                                                             config2.pop_size)
            self.species2 = config2.species_set_type(config2.species_set_config, self.reporters2)
            self.species2.speciate(config2, self.population2, self.generation)
        else:
            self.population1, self.species1, self.population2, self.species2, self.generation = initial_state

        self.best_genome1 = None
        self.best_genome2 = None

    def add_reporter(self, reporter, index=-1):
        if index == 0:
            self.reporters1.add(reporter)
        elif index == 1:
            self.reporters2.add(reporter)
        else:
            self.reporters1.add(reporter)
            self.reporters2.add(reporter)

    def remove_reporter(self, reporter, index=-1):
        if index == 0:
            self.reporters1.remove(reporter)
        elif index == 1:
            self.reporters2.remove(reporter)
        else:
            self.reporters1.remove(reporter)
            self.reporters2.remove(reporter)

    async def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        complete1 = False
        complete2 = False

        if (self.config1.no_fitness_termination or self.config2.no_fitness_termination) and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n and not (complete1 and complete2):
            k += 1

            self.reporters1.start_generation(self.generation)
            self.reporters2.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            await fitness_function(list(iteritems(self.population1)), list(iteritems(self.population2)), self.config1, self.config2)

            # Gather and report statistics.
            best1 = None
            for g in itervalues(self.population1):
                if best1 is None or g.fitness > best1.fitness:
                    best1 = g
            self.reporters1.post_evaluate(self.config1, self.population1, self.species1, best1)
            best2 = None
            for g in itervalues(self.population2):
                if best2 is None or g.fitness > best2.fitness:
                    best2 = g
            self.reporters2.post_evaluate(self.config2, self.population2, self.species2, best2)

            # Track the best genome ever seen.
            if self.best_genome1 is None or best1.fitness > self.best_genome1.fitness:
                self.best_genome1 = best1
            if self.best_genome2 is None or best2.fitness > self.best_genome2.fitness:
                self.best_genome2 = best2

            if not self.config1.no_fitness_termination or not self.config2.no_fitness_termination:

                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population1))
                if fv >= self.config1.fitness_threshold:
                    self.reporters1.found_solution(self.config1, self.generation, best1)
                    complete1 = True

                if fv >= self.config2.fitness_threshold:
                    self.reporters2.found_solution(self.config2, self.generation, best2)
                    complete2 = True

            # Create the next generation from the current generation.
            self.population1 = self.reproduction1.reproduce(self.config1, self.species1,
                                                            self.config1.pop_size, self.generation)
            self.population2 = self.reproduction2.reproduce(self.config2, self.species2,
                                                            self.config2.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species1.species:
                self.reporters1.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config1.reset_on_extinction:
                    self.population1 = self.reproduction1.create_new(self.config1.genome_type,
                                                                     self.config1.genome_config,
                                                                     self.config1.pop_size)
                else:
                    raise CompleteExtinctionException()
            if not self.species2.species:
                self.reporters2.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config2.reset_on_extinction:
                    self.population2 = self.reproduction2.create_new(self.config2.genome_type,
                                                                     self.config2.genome_config,
                                                                     self.config2.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species1.speciate(self.config1, self.population1, self.generation)
            self.reporters1.end_generation(self.config1, self.population1, self.species1)

            self.species2.speciate(self.config2, self.population2, self.generation)
            self.reporters2.end_generation(self.config2, self.population2, self.species2)

            self.generation += 1

        if self.config1.no_fitness_termination:
            self.reporters1.found_solution(self.config1, self.generation, self.best_genome1)
        if self.config2.no_fitness_termination:
            self.reporters2.found_solution(self.config2, self.generation, self.best_genome2)

        return self.best_genome1, self.best_genome2
