import numpy as np
import random


class KMeansHelper:
    # nrClusters: in parameter Number k clusters
    # numberIterations: How often k Means will be called
    def __init__(self, nrClusters, numberIterations):
        self.nrClusters = nrClusters
        self.numberOfIterations = numberIterations

    # data numpy Array containing the data Kmeans should be run on
    def runKMeans(self, data):
        realAssignment = []
        realLow = 99999999

        for i in range(0, self.numberOfIterations):
            tempClusters = (random.sample(data, self.nrClusters))
            curResult = self.runAlgorithm(data, tempClusters)
            if curResult[1] < realLow:
                realLow = curResult[1]
                realAssignment = curResult[0]

        return realAssignment

    # data numpy Array containing data
    # tempClusters numpy Array containing the initial values for clusters
    def runAlgorithm(self, data, tempClusters):
        lastAssignment = np.arange(0, len(data))
        curClusters = tempClusters

        while(True):
            curAssignment = lastAssignment
            for i in range(0, len(data)):
                curLow = sum((data[i] - curClusters[0])**2)
                curIndex = 0
                for k in range(1, self.nrClusters):
                    kLow = sum((data[i] - curClusters[k])**2)
                    if kLow < curLow:
                        curLow = kLow
                        curIndex = k

                curAssignment[i] = curIndex

            for k in range(0, self.nrClusters):
                t = (curAssignment == k)
                curClusters[k] = sum(data[t])/(max([1,  sum(t)]))

            if(sum(lastAssignment == curAssignment) == len(data)):
                break

        return (lastAssignment, self.cost(data, curClusters, lastAssignment))

    # data numpy Array containing data
    # clusters numpy array containig values for clusters
    # assignment the resulting assignment for those clusters
    def cost(self, data, clusters, assignment):
        result = 0
        for i in range(0, len(data)):
            result += sum((data[i] - clusters[assignment[i]])**2)

        return result
