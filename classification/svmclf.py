import numpy as np
from sklearn import svm

def svmclassify(training_set, training_class, testing_set, testing_class, verbose=True):

    #safeChecks
    if len(training_set)!=len(training_class):
        raise ValueError("The training set and training class must have the same length, "
                         "training set has a length of %d and training class has a length of %d"
                         %(len(training_set), len(training_class)))
    if len(testing_set)!=len(testing_class):
        raise ValueError("The testing set and testing class must have the same length, "
                         "testing set has a length of %d and testing class has a length of %d"
                         %(len(testing_set), len(testing_class)))

    #Number of classes for effectiveness analysis
    #Setting up classifier
    clf = svm.LinearSVC()

    #Training classifier with provided data set+group
    clf.fit(training_set, training_class)

    classifications = clf.score(testing_set, testing_class)

    print ("Correct classifications", classifications)
    #testing_class #
    #Setup for control
    # check, error = 0 , 0
    # error_matrix = np.zeros((classCount,classCount))
    #
    # #For each testing element
    # for i in range (len(testing_set)):
    #
    #     #Get prediction
    #     prediction = clf.predict(testing_set[i])
    #
    #     error_matrix[testing_class[i] - 1, prediction - 1]+=1
    #     if (verbose==1):
    #         print ("Predicting item {} , {} as {}".format(i, testing_class[i], prediction))
    #     if (testing_class[i] == prediction):
    #         check+=1
    #     else:
    #         error+=1
    #
    # return (check, error, error_matrix)
