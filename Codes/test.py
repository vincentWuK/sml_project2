import numpy as np
import keras
import pandas as pd
import ijson
from dataSelection import read_test_data


def predict_deep(X_test, model):
    pre = model.predict(X_test)
    return pre


testX = read_test_data()
model = keras.models.load_model('model30epoch')
predict = predict_deep(testX, model)
predict = np.round(predict)
labels = []
for row, p in enumerate(predict):
    authors = ""
    for ind, v in enumerate(p):
        if v == 1:
            authors = authors + str(ind) + " "
    authors = authors[:-1]
    labels.append(authors)
for indx, v in enumerate(labels):
    if v == "":
        labels[indx] = "-1"
id = [i for i in range(len(labels))]
result = {"ID": id, "Predict": labels}
df = pd.DataFrame.from_dict(result)
df.to_csv("../Result/test.csv", index=False)
