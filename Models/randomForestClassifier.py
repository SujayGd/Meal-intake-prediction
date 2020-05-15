import sys
sys.path.append('../')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.datasets import load_digits
import timeit
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.externals import joblib 

from helper import get_meal_vectors

x, y = get_meal_vectors('randomForestClassifier',apply_pca=True, padding=True, load_pca=False)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

kf = KFold(n_splits=10, shuffle=True)

clf = RandomForestClassifier(bootstrap=False, class_weight=None,
                                        criterion='gini', max_depth=None,
                                        max_features=0.05, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=5, min_samples_split=6,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=100, n_jobs=None,
                                        oob_score=False, random_state=23,
                                        verbose=0, warm_start=False)
                                        
f1scores = []
accscores = []
precscores = []
recallscores = []

for train_index, test_index in kf.split(x):
    train_data, test_data = x[train_index], x[test_index]
    train_labels, test_labels = y[train_index], y[test_index]

    clf.fit(train_data,train_labels)
    y_pred=clf.predict(test_data)
    f1scores.append(f1_score(test_labels, y_pred))
    accscores.append(accuracy_score(test_labels, y_pred))
    precscores.append(precision_score(test_labels, y_pred))
    recallscores.append(recall_score(test_labels, y_pred))

# Print the scores
print("F1 Scores: " , str(round(np.mean(f1scores)*100, 2)),"%")
print("Accuracy Scores: " ,str(round(np.mean(accscores)*100, 2)),"%") 
print("Precision Scores: " ,str(round(np.mean(precscores)*100, 2)),"%") 
print("Recall Scores: " , str(round(np.mean(recallscores)*100, 2)),"%") 

clf.fit(x,y)

saved_model = joblib.dump(clf, 'randomForestClassifier.pkl')



