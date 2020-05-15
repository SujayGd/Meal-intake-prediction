import sys
sys.path.append('../')
from helper import get_meal_vectors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.externals import joblib 


data, labels = get_meal_vectors('multiLayerPerceptronClassifier',True, False, False)

if __name__ == "__main__":

    # Initialise KFold with number of splits = 20 and random handling of data through shuffle
    kf = KFold(n_splits=20, shuffle=True)
    
    # Train_index, test_index in kf.split(data):
    k = MLPClassifier()
    
    # Intitalising scores 
    f1scores = []
    accscores = []
    precscores = []
    recallscores = []
    
    # Split Data and calculate test data, train data and test labels and train labels.
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
    
    # Fit the data, normalise it.
        k.fit(train_data, train_labels)
        pred_labels = k.predict(test_data)
    
    # Store the scores
        f1scores.append(f1_score(test_labels, pred_labels))
        accscores.append(accuracy_score(test_labels, pred_labels))
        precscores.append(precision_score(test_labels, pred_labels))
        recallscores.append(recall_score(test_labels, pred_labels))

    # Print the scores for Kfold cross validation
    print("Score for MLP")
    print("F1 Scores: " , str(round(np.mean(f1scores)*100, 2)),"%")
    print("Accuracy Scores: " ,str(round(np.mean(accscores)*100, 2)),"%") 
    print("Precision Scores: " ,str(round(np.mean(precscores)*100, 2)),"%") 
    print("Recall Scores: " , str(round(np.mean(recallscores)*100, 2)),"%") 

    #fit all the data
    k.fit(data, labels)
    joblib.dump(k, 'multiLayerPerceptronClassifier.pkl')

