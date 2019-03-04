# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array




"""
Commandline parameter(s):
    none
"""

BASE_PATH = os.getcwd() + "/"
OUTFILE_BASE = BASE_PATH + 'out/tsp/'

if not os.path.exists(OUTFILE_BASE):
    os.makedirs(OUTFILE_BASE)

for algo in ["RHC","MIMIC","GA","SA"]:
    with open(OUTFILE_BASE+algo+".csv", 'w') as f:
        f.write("iterations,training_time,fitness\n")

# set N value.  This is the number of points
# N = 50
for N in xrange(2, 1000):
    random = Random()

    points = [[0 for x in xrange(2)] for x in xrange(N)]
    for i in range(0, len(points)):
        points[i][0] = random.nextDouble()
        points[i][1] = random.nextDouble()

    ef = TravelingSalesmanRouteEvaluationFunction(points)
    odd = DiscretePermutationDistribution(N)
    nf = SwapNeighbor()
    mf = SwapMutation()
    cf = TravelingSalesmanCrossOver(ef)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))

    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "RHC")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(rhc.getOptimal())))

    sa = SimulatedAnnealing(1E12, .999, hcp)
    fit = FixedIterationTrainer(sa, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))

    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "SA")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(sa.getOptimal())))

    ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
    fit = FixedIterationTrainer(ga, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))

    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "GA")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(ga.getOptimal())))

    # for mimic we use a sort encoding
    ef = TravelingSalesmanSortEvaluationFunction(points);
    fill = [N] * N
    ranges = array('i', fill)
    odd = DiscreteUniformDistribution(ranges);
    df = DiscreteDependencyTree(.1, ranges);
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

    mimic = MIMIC(500, 100, pop)
    fit = FixedIterationTrainer(mimic, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))

    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "MIMIC")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(mimic.getOptimal())))
