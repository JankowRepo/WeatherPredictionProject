from keras import Sequential, layers
from keras.layers import Dense, Conv1D, Flatten
from keras.optimizers import SGD
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
import tensorflow as tf
from keras import backend as K


def TreeGradientBoosting(X,y):
    model = LGBMClassifier()
    model.fit(X, y)
    return model


def LinearRegressionModel(X,y):
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


def LogisticRegressionModel(X,y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def DecisionTree(X,y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


def SupportVectorRegression(X,y):
    model = SVR()
    model.fit(X, y)
    return model

def NeuralNetworks(X,y):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=0,
        patience=10,
        mode='auto',
        restore_best_weights=True)

    epochs=100
    batch_size=8


    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(layers.Dropout(0.4))

    model.add(Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(Dense(16, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(Dense(4, activation="relu"))
    model.add(layers.Dropout(0.33))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.008, momentum=0.85),
                  metrics=[f1, 'accuracy'])

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y)

    model.fit(X_tr, y_tr, epochs=epochs, verbose=True, batch_size=batch_size, callbacks=[early_stopping],
              validation_data=(X_val, y_val))

    return model

def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val