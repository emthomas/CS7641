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
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
from array import array


BASE_PATH = os.getcwd() + "/"
OUTFILE_BASE = BASE_PATH + 'out/countones/'

if not os.path.exists(OUTFILE_BASE):
    os.makedirs(OUTFILE_BASE)

for algo in ["RHC","MIMIC","GA","SA"]:
    with open(OUTFILE_BASE+algo+".csv", 'w') as f:
        f.write("iterations,training_time,fitness\n")

for N in xrange(10, 1000):
    fill = [2] * N
    ranges = array('i', fill)

    ef = CountOnesEvaluationFunction()
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "RHC: " + str(ef.value(rhc.getOptimal()))
    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "RHC")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(rhc.getOptimal())))

    sa = SimulatedAnnealing(1E11, .95, hcp)
    fit = FixedIterationTrainer(sa, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "SA: " + str(ef.value(sa.getOptimal()))
    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "SA")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(sa.getOptimal())))

    ga = StandardGeneticAlgorithm(200, 100, 10, gap)
    fit = FixedIterationTrainer(ga, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "GA: " + str(ef.value(ga.getOptimal()))
    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "GA")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(ga.getOptimal())))

    mimic = MIMIC(200, 20, pop)
    fit = FixedIterationTrainer(mimic, N)
    start = time.time()
    fit.train()
    end = time.time()
    training_time = end - start
    print "MIMIC: " + str(ef.value(mimic.getOptimal()))
    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, "MIMIC")
    with open(OUTFILE, 'a+') as f:
        f.write("%d,%f,%f\n" % (N, training_time, ef.value(mimic.getOptimal())))
