import numpy as np
import random


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        # "+1" -> add an additional bias neuron to the input layer to provide a controlled input correction.
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-1.0, 1.0)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-1.0, 1.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def forward(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for h in range(self.hidden_n):
            total_h = 0.0
            for i in range(self.input_n):
                total_h += self.input_cells[i] * self.input_weights[i][h]
            self.hidden_cells[h] = sigmoid(total_h)
        # activate output layer
        for o in range(self.output_n):
            total_o = 0.0
            for h in range(self.hidden_n):
                total_o += self.hidden_cells[h] * self.output_weights[h][o]
            self.output_cells[o] = sigmoid(total_o)
        return self.output_cells[:]

    def backward(self, case, label, learn, correct):
        # feed forward
        self.forward(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error_o = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error_o
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error_h = 0.0
            for o in range(self.output_n):
                error_h += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error_h
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change_o = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change_o + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change_o
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change_i = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change_i + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change_i
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.backward(case, label, learn, correct)
                print("cases:", i, "training steps:", j, "loss:", error)

    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05, 0.1)
        for i in range(4):
            print("case:", cases[i], "label:", labels[i], "output:", self.forward(cases[i]))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()

# print(make_matrix(5, 4))


# 820.1, 405.2
# 130.1, 270.2
# 610.1, 720.3
# 1490.1, 450.1
# 110.1, 465.3
# 290.1, 345.5
# 540.1, 55.5
# 540.1, 240.6
# 540.1, 625.6
# 830.1, 475.4
# 900.1, 100.5
# 1030.1, 7404
# 1110.1, 345.5
# 1320.1, 465.1


# for i in range(50):
#     print(i)
#     if i == 10:
#         print("ok")
#         break
