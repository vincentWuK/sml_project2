import numpy as np
from sklearn.metrics import f1_score
from dataSelection import read_data


def deep_model(feature_dim, label_dim):
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    model.add(Dense(1000, activation='elu', input_dim=feature_dim))
    # model.add(tf.keras.layers.Dropout(0.1)),
    model.add(Dense(512, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    model.compile(optimizer='RMSprop', loss="binary_crossentropy", metrics=['accuracy'])
    return model


def train_deep(X_train, y_train, X_test, y_test):
    feature_dim = X_train.shape[1]
    label_dim = y_train.shape[1]
    model = deep_model(feature_dim, label_dim)
    model.summary()
    model.fit(X_train, y_train, batch_size=512, epochs=30, validation_data=(X_test, y_test))
    return model


def predict_deep(X_test, model):
    pre = model.predict(X_test)
    return pre


def print_acc(X, Y, model):
    ppred = predict_deep(X, model)
    a = np.round(ppred)
    f1 = f1_score(Y, a, average='weighted')
    print("f1 score: ", f1)
    a = np.round(ppred)
    c = 0
    b = Y
    for ind, p in enumerate(a):
        if all(p == b[ind]):
            c += 1
    print("acc: ", c / len(a))


data = read_data()
X = data[0]
Y = data[1]
train_X = X[0:6000]
train_Y = Y[0:6000]
test_X = X[6000::]
test_Y = Y[6000::]
model = train_deep(train_X, train_Y, test_X, test_Y)
print("test training...")
print_acc(train_X, train_Y, model)
print("test validation...")
print_acc(test_X, test_Y, model)

model.save("model30epoch")
