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
    kmeansHelper = KMeansHelper(3, 150)
    dataset = datasets.load_iris()
    data = np.array(dataset["data"])

    realResults = np.array(dataset["target"])
    # plotData(data,0)
    kmeansHelper.runKMeans(data)
    myResult = kmeansHelper.runKMeans(data)
    succeed = 0
    for i in range(0, len(myResult)):
        if myResult[i] == realResults[i]:
            succeed += 1
        else:
            print(i, "fail")
    print("Success Rate: ", succeed*1.0/len(data))
if __name__ == "__main__":
    main()
