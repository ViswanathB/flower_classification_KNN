#Import libraries

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
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

def loaddataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    try:
        dataset = pandas.read_csv(url, names=names)
        return dataset
    except:
        print("Error while reading dataset")

def classify(train_input,train_output,test_input,expected_output):
    knn = KNeighborsClassifier()
    #print(train_input)
    #print("Train output:")
    #print(train_output)
    knn.fit(train_input,train_output)
    observed_output = knn.predict(test_input)
    print("Accuracy: %f" % accuracy_score(observed_output, expected_output))
    print(classification_report(observed_output, expected_output))

if __name__ == '__main__':
    dataset = loaddataset()
    #print(dataset.groupby('class').size())
    #print(dataset)
    array = dataset.values

    Attributes = array[:,0:4]
    R = array[:,4]
    validation_size = 0.20
    seed = 7
    train_input, test_input, train_output, test_output = model_selection.train_test_split(Attributes, R, test_size=validation_size, random_state=seed)
    classify(train_input,train_output,test_input,test_output)
