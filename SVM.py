import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm

dataPath = "HW3_Data/ecs171.dataset.txt"
data = pd.read_table(dataPath)

class SVM(object):
    Xtrain = None
    ytrain = None
    Xtest = None
    ytest = None
    kernel = 'linear'
    model = None
    yPredict = None
    score = None

    def __init__(self, xtrain, ytain, xtest, ytest):
        self.Xtrain = xtrain
        self.Ytrain = ytain
        self.Xtest = xtest
        self.ytest = ytest

    def setKernel(self, value):
        self.kernel = value

    def createModel(self):
        self.model = svm.SVC(kernel=self.kernel)

    def fitModel(self):
        self.model.fit(self.Xtrain, self.Ytrain)

    def predict(self):
        self.yPredict = self.model.predict(self.Xtest)
        self.score = accuracy_score(self.yPredict, self.ytest, normalize=True)


def non_zero_coeff(X, y, _lambda):
    clf = Ridge(normalize=True, solver='sag')
    # clf = Lasso()
    clf.set_params(alpha=_lambda)
    clf.fit(X, y)
    nonzero = np.count_nonzero(clf.coef_)
    print("Number of non-zero features: " + str(nonzero) + ", alpha: " + str(_lambda))
    return nonzero, (clf.coef_ != 0)

def svm_cross_validation(features, predictors):
    return 0

def main():
    X = data.iloc[:, 6:].values
    for predictor in ["Strain", "Medium", "Stress", "GenePerturbed"]:
        le = LabelEncoder()
        y = le.fit_transform(data[predictor].values)
        non, cols = non_zero_coeff(X, y, 0.9)
        non_zero_cols = data.iloc[:, 6:].loc[:, cols].columns.values.tolist()
        Features = data[non_zero_cols].values
        svm_cross_validation(Features, y)

if __name__ == '__main__':
    main()