import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For graphical representation 
import seaborn as sns # Python visualization library based on matplotlib provides a high-level interface for drawing attractive statistical graphics
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedKFold
import csv
from sklearn.model_selection import train_test_split
#skfolds=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#def models(X_train, Y_train,score):
#    clfs = []
#    result = []
#    names = []
#    clfs.append(('LR', LogisticRegression()))
#    clfs.append(('LDA', LinearDiscriminantAnalysis()))
#    clfs.append(('KNN', KNeighborsClassifier()))
#    clfs.append(('CART', DecisionTreeClassifier()))
#    clfs.append(('NB', GaussianNB()))
#    clfs.append(('SVM', SVC()))
#    for algo_name, clf in clfs:
#        k_fold = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)
#        cv_score = model_selection.cross_val_score(clf, X_train, Y_train, cv=k_fold, scoring=score)
#        #result = "%s: %f (%f)" % (algo_name, cv_score.mean(), cv_score.std())
#        result.append((algo_name,cv_score.mean(), cv_score.std()))
#        names.append(algo_name)
#    return (result)


#print(check_output(["ipconfig"]).decode("utf-8"))

#print(sepal_length)
#print(variety)
with open('input/iris.csv', newline='') as csvfile:
    csv_array=[]
    rows = csv.reader(csvfile)
    for row in rows:
        csv_array.append(row)
        #print(row)

def spl_list(array,n):
    for i in range(0,len(array),n):
        yield array[i:i+n]

#print(csv_array)
train_data_array=[]
train_lable_array=[]
for i in range(1,len(csv_array)-1,1):
    single_csv_data=csv_array[i]
    train_data_array.append(single_csv_data[0])
    train_data_array.append(single_csv_data[1])
    train_data_array.append(single_csv_data[2])
    train_data_array.append(single_csv_data[3])
    train_lable_array.append(single_csv_data[4])
print(train_data_array)
for i in train_data_array:
    print(i)

train_data_array_new=list(spl_list(train_data_array,4))
train_data = train_data_array_new
train_lable = train_lable_array 
knn = KNeighborsClassifier()
train_data, test_data, train_lable, test_lable = train_test_split(train_data, train_lable, test_size=0.3)
knn.fit(train_data,train_lable)
predict_result=knn.predict(test_data)
#print(predict_result)
final_array=[]
for i in predict_result:
    #print(i)
    final_array.append(i)
#print(len(final_array))
#print(len(test_lable))
#print(test_lable)
detect_count=0
for num in range(0,len(test_lable)-1,1):
    final_lable=final_array[num]
    test_lable2=test_lable[num]
    #print(final_lable)
    #print(test_lable2)
    if final_lable==test_lable2:
        detect_count=detect_count+1
aim_per=detect_count/len(test_lable)
print(f"命中率: {aim_per}")
