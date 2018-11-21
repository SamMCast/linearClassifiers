import numpy as np 
from matplotlib import pyplot as plt

class LinearClassifier:

    def __isMisclasified(self,x,y, w):
        """
        checks if the predicted value label is misclassified
        x:  data point to be processed (vector)
        y:  the correct data label 
        w:  the weights vector
        Returns: True if the y*(w^T)x < 1 and false otherwise
        """
        return y * np.dot(w,x) < 1

    #misclassified w =w + n(yx - 2lambdaw) ywx < 1
    #correct w = n(-2lambaw) 
    def svm_sgd_plot(self, features, labels):
        """
        performs sochastic gradient descent to determine our model
        features: our data set
        labels: the classes we assigned to each data entry
        Returns: our learned model
        """
        #initialize our weight vector with random values (3 values)
        w = np.random.rand(len(features[0]))
        #The learning rate
        eta = 1
        #how many iterations to train for
        epochs = 100000
        #store misclassifications so we can plot how they change over time
        errors = []

        #training part, gradient descent part
        for epoch in range(1,epochs):
            error = 0
            for ind, x in enumerate(features):
                #misclassification
                if (self.__isMisclasified(x,labels[ind],w)):
                    w = w + eta *((labels[ind]* x )+(-2*(1/epoch)*w))
                    error =1 
                else:
                    #correct classification, update our weights
                    w = w + eta *(-2* (1/epoch)*w)
            errors.append(error)
        

        #Lets plot the rate of classification errors during training for our SVM
        plt.plot(errors, "|")
        plt.ylim(0.5,1.5)
        plt.axes().set_yticklabels([])
        plt.xlabel("Epoch")
        plt.ylabel("Misclassified")
        plt.show()

        return w







