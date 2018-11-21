#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
from sochastic_gradient_descent import LinearClassifier as LM

def main():
    #input data -of the form [X value, Y value, Bias term]
    X = np.array([[-2,4,-1], [4,1,-1], [1,6,-1],[2,4,-1],[6,2,-1]])

    #Associated output labels  -First 2 examples are labeled "-1" and last 3
    y = np.array([-1,-1,1,1,1])

    #lets plot these examples on a 2D graph!
    #for each example
    for d, sample in enumerate(X):
        #plot the negative samples (the first 2)
        if d < 2:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
        #Plot the positive samples (the last 2)
        else:
            plt.scatter(sample[0], sample[1], s=120, marker="+", linewidth=2)

    #Print a possible hyperplane, that is separating the two classes.
    #we'll two points and draw the line between them(naive guess)
    plt.plot([-2,6],[6,0.5])
    plt.show()

    SVM = LM()

    w = SVM.svm_sgd_plot(X,y)

    # Add our test samples
    plt.scatter(2,2, s=120, marker = "_", linewidths=2, color="yellow")
    plt.scatter(4,3, s=120, marker = "+", linewidths=2, color="blue")

    #Print the hyperplane calculated by svm_sdg()
    x2 = [w[0], w[1], -w[1], w[0]]
    x3 = [w[0], w[1], w[1], -w[0]]

    x2x3 = np.array([x2,x3])

    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y, U,V, scale=1, color="blue")

    plt.show()

main()
