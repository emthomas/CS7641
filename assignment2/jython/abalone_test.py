"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

RAW_INPUT_FILE = os.path.join("..", "src", "opt", "test", "abalone.txt")
TEST_INPUT_FILE = os.path.join("..", "src", "opt", "test", "abalone_test.csv")
TRAIN_INPUT_FILE = os.path.join("..", "src", "opt", "test", "abalone_train.csv")
BASE_PATH = os.getcwd() + "/"
OUTFILE_BASE = BASE_PATH + 'out/'

INPUT_LAYER = 7
HIDDEN_LAYER = 13
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1001


def initialize_instances():
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []
    instances_test = []

    # Read in the abalone_train.csv CSV file
    with open(TRAIN_INPUT_FILE, "r") as abalone:
        reader = csv.reader(abalone)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) < 15 else 1))
            instances.append(instance)

    # Read in the abalone_train.csv CSV file
    with open(TEST_INPUT_FILE, "r") as abalone:
        reader = csv.reader(abalone)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) < 15 else 1))
            instances_test.append(instance)

    return instances, instances_test


def train(oa, network, oaName, instances, measure, test_instances):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """

    if not os.path.exists(OUTFILE_BASE):
        os.makedirs(OUTFILE_BASE)

    OUTFILE = "%s%s.csv" % (OUTFILE_BASE, oaName)

    print "\nError results for %s\n---------------------------" % (oaName,)
    print "iterations,training_time,train_MSE,train_acc,test_MSE,test_acc"

    with open(OUTFILE, 'w') as f:
        f.write('iterations,training_time,train_MSE,train_acc,test_MSE,test_acc\n')

    for iteration in xrange(TRAINING_ITERATIONS):
        start = time.time()
        oa.train()
        end = time.time()
        training_time = end - start


        N = len(instances)
        error = 0.00
        correct = 0
        incorrect = 0
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            actual = instance.getLabel().getContinuous()
            predicted = network.getOutputValues().get(0)
            predicted = max(min(predicted, 1), 0)
            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        train_MSE = error / float(N)
        train_acc = correct / float(correct + incorrect)

        optimal_instance = oa.getOptimal()
        network.setWeights(optimal_instance.getData())

        N = len(test_instances)
        error = 0.00
        correct = 0
        incorrect = 0
        for instance in test_instances:
            network.setInputValues(instance.getData())
            network.run()

            actual = instance.getLabel().getContinuous()
            predicted = network.getOutputValues().get(0)
            predicted = max(min(predicted, 1), 0)
            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        MSE = error / float(N)
        acc = correct / float(correct + incorrect)

        if iteration % 1 == 0:
            with open(OUTFILE, 'a+') as f:
                f.write("%d,%f,%f,%f,%f,%f\n" % (iteration, training_time, train_MSE, train_acc, MSE, acc))
            print "%d,%f,%f,%f,%f,%f" % (iteration, training_time, train_MSE, train_acc, MSE, acc)


def main():
    """Run algorithms on the abalone dataset."""
    train_instances, test_instances = initialize_instances()
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    for i, name in enumerate(oa_names):
        start = time.time()

        train(oa[i], networks[i], oa_names[i], train_instances, measure, test_instances)
        end = time.time()
        training_time = end - start

    print results


if __name__ == "__main__":
    main()

