import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Helper.KMeansHelper import KMeansHelper


def plotData(data):
    l = []
    for i in range(0, len(data)):
        l.append(data[i][0])
    plt.plot(l)
    l2 = []
    for i in range(0, len(data)):
        l2.append(data[i][1])

    plt.plot(l2)


def plotAssignment(data, assignment, indexes):
    for i in indexes:
        plt.subplot(2, 2, i + 1)
        for j in range(0, 3):
            l = []
            for idx in range(0, len(data)):
                if assignment[idx] == j:
                    l.append(data[idx][i])

            if j == 0:
                plt.plot(l, color='r', marker='o', linestyle='None')
            elif j == 1:
                plt.plot(l, color='g', marker='o', linestyle='None')
            else:
                plt.plot(l, marker='o', color='b', linestyle='None')


def main():
    kmeansHelper = KMeansHelper(3, 150)
    dataset = datasets.load_iris()
    data = np.array(dataset["data"])

    realResults = np.array(dataset["target"])
    # plotData(data,0)
    kmeansHelper.runKMeans(data)
    myResult = kmeansHelper.runKMeans(data)
    succeed = 0

    indexDict = {}

    for j in range(0, 3):
        l = [0, 0, 0]
        for i in range(0, len(myResult)):
            if realResults[i] == j:
                l[myResult[i]] += 1
        idx = l.index(max(l))
        indexDict[idx] = j

    for i in range(0, len(myResult)):
        if indexDict[myResult[i]] == realResults[i]:
            succeed += 1
        else:
            print(i, myResult[i], realResults[i], "fail")
    plotAssignment(data, myResult, [0, 1, 2, 3])
    plt.show()
    print("Success Rate: ", succeed*1.0/len(data))

if __name__ == "__main__":
    main()
