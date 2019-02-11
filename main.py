from ANN import run as run_ann
from Boosting import run as run_boosting
from DecisionTree import run as run_dt
from KNN import run as run_knn
from SVM import run as run_svm
import pandas as pd

from helpers import plot_distribution

if __name__ == "__main__":
    dataset = pd.read_csv('./data/wine/winequality-white.csv', sep=",")
    plot_distribution(dataset, 'quality', 'Distribution of Quality', 'Quality', 'Count', 9).show()

    dataset = pd.read_csv('./data/abalone/abalone_data.csv', sep=',')
    plot_distribution(dataset, 'rings', 'Distribution of Age', 'Age', 'Count', 20).show()

    run_ann()
    run_boosting()
    run_dt()
    run_knn()
    run_svm()

