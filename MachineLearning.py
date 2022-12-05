from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek

def GBM(X,y):
    model = LGBMClassifier()
    model.fit(X, y)
    return model