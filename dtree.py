# Decision Tree Implementation

import numpy as np


class dtree:
    """ A basic Decision Tree"""

    def __init__(self):
        """ Constructor """

    def read_data(self, filename):
        """Read data in and save them into three variables
        data: store all data for the features (class excluded)
        classes: store all data for the class
        featureNames: stores all feature names
        featureValues: store each feature's unique values
        """

        fid = open(filename, "r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(","))  # list of lists
        fid.close()

        self.featureNames = data[0]  # first row as feature names
        self.targetName = self.featureNames[-1]
        self.featureNames = self.featureNames[:-1]

        data = data[1:]  # remove the first row
        self.classData = []
        for d in range(len(data)):
            self.classData.append(data[d][-1])  # extract last column
            data[d] = data[d][:-1]  # remove the last column in data

        # extract unique values values for each feature
        transposedData = np.transpose(np.copy(data))
        self.featureValues = {}
        for i in range(len(self.featureNames)):
            self.featureValues[self.featureNames[i]] = np.unique(transposedData[i])
        print self.featureValues

        return data, self.classData, self.featureNames

    def ID3(self, data, classData, featureNames, parentMajority):

        """ The ID3 algorithm, which recursively constructs the tree"""

        nData = len(data)
        nClasses = len(classData)

        # base case 1: if D is empty, return the parentMajority class
        if nData == 0 and nClasses == 0:
            return parentMajority

        # get the number of features
        nFeatures = 0
        if nData != 0:
            nFeatures = len(data[0])

        # find the majority of target value
        majority = self.majority_class(classData)

        # base case 2: if d is empty (no features), return the majority class
        if nFeatures == 0:
            return majority

        # base case 3: if all instances have the same target value, return the first target value
        elif classData.count(classData[0]) == nData:
            return classData[0]

        # general case to recursively build the tree
        else:

            # Choose the best feature based on information gain
            gain = np.zeros(nFeatures)
            for feature in range(nFeatures):
                gain[feature] = self.info_gain(data, classData, feature)
            bestFeature = np.argmax(gain)
            bestFeatureName = featureNames[bestFeature]

            tree = {bestFeatureName: {}}
            # print "The tree %s afer the best feature %s" % (tree, bestFeatureName)

            # Load the bestFeature's possible values into a list
            values = []
            for i in range(len(self.featureValues[bestFeatureName])):
                values.append(self.featureValues[bestFeatureName][i])
            # print "The best feature %s values %s" % (bestFeatureName, str(values))

            # Partition the original datapoints based on the best feature possible values
            # and then recursively invoke ID algorithm to build subtrees
            for value in values:
                newData = []
                newClassData = []
                index = 0

                # partition the data
                for datapoint in data:
                    if datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            newdatapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature == nFeatures:
                            newdatapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            newdatapoint = datapoint[:bestFeature]
                            newdatapoint.extend(datapoint[bestFeature + 1:])
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature + 1:])

                        newData.append(newdatapoint)
                        newClassData.append(classData[index])
                    index += 1

                # Now do recursive call to build the subtrees
                subtree = self.ID3(newData, newClassData, newNames, majority)

                # Add the subtree on to the tree
                # print "The subtree %s for the current tree %s" % ( subtree, tree,)
                tree[bestFeatureName][value] = subtree

            return tree

    def classify(self, tree, datapoint):
        """ classfication on a new datapoint using a tree"""

        if type(tree) == type("string"):
            return tree
        else:
            a = tree.keys()[0]
            for i in range(len(self.featureNames)):
                if self.featureNames[i] == a:
                    break

            try:
                t = tree[a][datapoint[i]]
                return self.classify(t, datapoint)
            except:
                return None

    def classifyAll(self, tree, data):
        """ classfication on a set of data using a tree"""

        results = []
        for i in range(len(data)):
            results.append(self.classify(tree, data[i]))
        return results

    def printTree(self, tree, str):
        """ print out the decision tree"""

        if type(tree) == dict:
            for item in tree.values()[0].keys():
                print "%s %s = %s " % (str, tree.keys()[0], item)
                self.printTree(tree.values()[0][item], str + "\t")
        else:
            print "%s -> %s = %s" % (str, self.targetName, tree)

    def entropy(self, classData):

        arr, counts = np.unique(classData, return_counts=True)
        classes = counts.astype('float') / len(classData)
        for clas in classes:
            if clas != 0.0:
                totalEntropy = clas * np.log2(clas)
            else:
                totalEntropy = 0
        return np.abs(totalEntropy)

    def info_gain(self, data, classData, featureIndex):
        """ Calculate informatin information"""
        ###### your implementation below ######
        gain = self.entropy(classData)
        rem = 0
        array = np.array(data)[0:, featureIndex]

        arr, counts = np.unique(array, return_counts=True)
        for i in range(len(arr)):
            partition = []
            weight = counts[i] / len(array)
            for j in range(len(array)):
                if array[j] == arr[i]:
                    partition.append(classData[j])
            rem += weight * self.entropy(partition)
        return gain - rem

    def majority_class(self, classData):
        """ find the majority of class"""

        ###### your implementation below ######
        arr, counts = np.unique(classData, return_counts=True)
        i = np.argmax(counts)
        return classData[i]

    def predictionAccuracy(self, predicted, actual):
        """ compute prediction accuracy on predicted data"""

        ###### your implementation below ######
        accuracy = 0
        for i in range(len(predicted)):
            if (predicted[i] == actual[i]):
                accuracy = accuracy + 1
        return (accuracy / len(actual)) * 100
