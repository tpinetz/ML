__author__ = 'Thomas'

from Node import Node


def identityFunction(x):
    return x


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.addLayer(1,identityFunction())

    def addLayer(self, numberOfNodes, costFunction):
        layer = []
        for i in range(0,numberOfNodes):
            n = Node(costFunction)
            layer.append(n)
        self.layers.append(layer)

    def feedForward(self, data):
        result = 0
        for layer in self.layers():
            result = 0

        return result

