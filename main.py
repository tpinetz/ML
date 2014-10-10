import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Helper.KMeansHelper import KMeansHelper


def main():
    kmeansHelper = KMeansHelper(3)
    dataset = datasets.load_iris()
    # kmeansHelper.runKMeans(dataset)
    data = np.array(dataset["data"])
    # print(dataset)
    plt.plot(data)
    plt.show()
    # for key in dataset:
        # print(dataset[key])

if __name__ == "__main__":
    main()
