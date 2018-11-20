#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt

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

main()
