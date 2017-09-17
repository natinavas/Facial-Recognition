import numpy as np

def euclassify(training_set, training_class, testing_set, testing_class, verbose=True):

    classCount = len(set(training_class))
    classCounter= np.zeros(classCount)

    matrix = np.zeros((classCount,len(training_set[0])))

    for k in range (len(training_set)):
        matrix[training_class[k] - 1]=np.add(matrix[training_class[k]-1],training_set[k])
        classCounter[training_class[k]-1] +=1

    meandist = matrix / classCounter

    for j in range (len(testing_set)):
        min= eudistance(testing_set[j],meandist[0])
        group= 1
        for i in range(1, len(classCount)):
            this = eudistance(testing_set[j],meandist[i])
            if (this<=min):
                min=this
                group = i+1
        print("Classified {} item from group {} as {}".format(j+1, testing_class[j], group))


def eudistance(x,y):
    if len(x)!= len(y):
        raise ValueError("wrong params")

    return np.sqrt(sum(np.square(x-y)))


