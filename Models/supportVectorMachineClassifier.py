import sys
sys.path.append('../')
from helper import get_meal_vectors
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB

from sklearn.externals import joblib


def getAccuracy(svmModel, testData, testGt):
    preds = svmModel.predict(testData)
    return f1_score(testGt, preds), accuracy_score(testGt, preds), precision_score(testGt, preds), recall_score(testGt, preds)

def trainSVM(trainingData, labels):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(trainingData)

    counter = 0
    f1List = []
    accuracyList = []
    presList = []
    recallList = []
    svclassifier = SVC(kernel='linear')
    for _ in range(1):
        for train_index, test_index in kf.split(trainingData):
            counter += 1
            svclassifier.fit(trainingData[train_index], labels[train_index])
            f1Score, acc, pres, rec = getAccuracy(svclassifier, trainingData[test_index], labels[test_index])
            f1List.append(f1Score)
            accuracyList.append(acc)
            presList.append(pres)
            recallList.append(rec)

    svclassifierTotal = SVC(kernel='linear')
    svclassifierTotal.fit(trainingData, labels)
    joblib.dump(svclassifierTotal, 'supportVectorMachineClassifier.pkl')

    print("-------------------------------------------------------------------")
    print("Accuracy: {}%".format(round(100 * sum(accuracyList)/len(accuracyList), 2)))
    print("F1: {}%".format(round(100 * sum(f1List) / len(f1List), 2)))
    print("Precision: {}%".format(round(100 * sum(presList) / len(presList), 2)))
    print("Recall: {}%".format(round(100 * sum(recallList) / len(recallList), 2)))


if __name__ == "__main__":
    data, labels = get_meal_vectors('supportVectorMachineClassifier')
    trainSVM(data, labels)