import matplotlib.pyplot as plt

class plotHelper:
    def __init__(self, data):
        self.data = data

    def plotData(self):
        l = []
        for i in range(0, len(self.data)):
            l.append(self.data[i][0])
        plt.plot(l)
        l2 = []
        for i in range(0, len(self.data)):
            l2.append(self.data[i][1])

        plt.plot(l2)

    def plotAssignment(self, assignment, indexes):
        for i in indexes:
            plt.subplot(2, 2, i + 1)
            for j in range(0, 3):
                l = []
                for idx in range(0, len(self.data)):
                    if assignment[idx] == j:
                        l.append(self.data[idx][i])

                if j == 0:
                    plt.plot(l, color='r', marker='o', linestyle='None')
                elif j == 1:
                    plt.plot(l, color='g', marker='o', linestyle='None')
                else:
                    plt.plot(l, marker='o', color='b', linestyle='None')

