import numpy as np
from sklearn import svm

def svmclassify(trainingSet, trainingClass,testingSet, testingClass, verbose):

    #safeChecks
    if (len(trainingSet)!=len(trainingClass) or  len(testingSet)!=len(testingClass)):
        #TODO pulir esto jeje
        print "invalid classification data"

    #Number of classes for effectiveness analysis
    classCount = len(set(trainingClass))

    #Setting up classifier
    clf = svm.SVC(gamma=0.001, C=100)

    #Training classifier with provided data set+group
    clf.fit(trainingSet,trainingClass)

    #Setup for control
    check, error = 0 , 0
    error_matrix = np.zeros((classCount,classCount))

    #For each testing element
    for i in range (len(testingSet)):
        #Get prediction
        prediction = clf.predict([testingSet[i]])

        error_matrix[testingClass[i]-1,prediction-1]+=1
        if (verbose==1):
            print ("Predicting item {} , {} as {}".format(i,testingClass[i],prediction))
        if ( testingClass[i] == prediction):
            check+=1
        else:
            error+=1

    return (check, error, error_matrix)

trGroup = np.array([1,1,1,2,2,2,3,3,3])
trSet=np.array([[1,1],[2,2],[1,1],[11,11],[12,12],[13,13],[45,45],[43,43],[44,44]])

testGroup= np.array([1,3,2])
testSet=np.array([[1,1],[44,44],[15,15]])

v = 1

right, wrong, matrix = svmclassify(trSet, trGroup, testSet, testGroup, v)

print("{} % correct predictions".format(right*100/(right+wrong)))

print(matrix)


