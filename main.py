import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Helper.KMeansHelper import KMeansHelper


def plotData(data, index):
    l = []
    for i in range(0, len(data)):
        l.append(data[i][index])
    plt.plot(l)
    plt.show()


def main():
    kmeansHelper = KMeansHelper(3)
    dataset = datasets.load_iris()
    data = np.array(dataset["data"])

    realResults = np.array(dataset["target"])
    # plotData(data,0)
    kmeansHelper.runKMeans(dataset)
    myResult = kmeansHelper.runKMeans(data)
    for i in range(0, len(myResult)):
        print(i + ": " +  (myResult == realResults))

if __name__ == "__main__":
    main()
