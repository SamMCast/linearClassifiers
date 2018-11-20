import numpy as np 

class LinearClassifier:

    def isMisclasified(self,x,y, w):
        """
        checks if the predicted value label is misclassified
        x:  data point to be processed (vector)
        y:  the correct data label 
        w:  the weights vector
        Returns: True if the y*(w^T)x < 1 and false otherwise
        """
        return y * np.dot(w,x) < 1

    #misclassified w =w + n(yx - 2w) ywx < 1
    #correct w = n(-2w) 
    def svm_sgd_plot(self, features, labels):
        #initialize our weight vector with random values (3 values)
        w = np.random.rand(len(features[0]))
        #The learning rate
        eta = 1
        #how many iterations to train for
        epochs = 100000
        #store misclassifications so we can plot how they change over time
        errors = []

        #training part, gradient descent part
        for epoch in range(epochs):
            for ind, x in enumerate(features):
                #misclassification
                if (self.isMisclasified(x,labels[ind],w)):
                    w = w + eta *(labels[ind]* x - (2*(1/epoch)*w))
                else:




