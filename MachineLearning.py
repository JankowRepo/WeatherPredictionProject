from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR


def TreeGradientBoosting(X,y):
    model = LGBMClassifier()
    model.fit(X, y)
    return model


def LinearRegression(X,y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def SupportVectorMachines(X,y):
    model = SVC()
    model.fit(X, y)
    return model


def KNN(X,y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model


def RandomForest(X,y):
    model = RandomForestClassifier(max_depth=3)
    model.fit(X, y)
    return model


def LogisticRegression(X,y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def DecisionTreeClassifier(X,y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


def SupportVectorRegression(X,y):
    model = SVR()
    model.fit(X, y)
    return model