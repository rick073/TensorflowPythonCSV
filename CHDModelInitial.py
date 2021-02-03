# For Google Collab
# try:
#  # %tensorflow_version only exists in Colab.
#  %tensorflow_version 2.x
# except Exception:
#  pass

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def main():
    tf.random.set_seed(1234)

    print("--Get data--")
    heartTrain = pd.read_csv("heartTrain.csv")
    heartTrainFeatures = heartTrain.copy()
    heartTrainLabels = heartTrainFeatures.pop("chd")
    heartTrainFeatures.pop("row.names")

    heartTest = pd.read_csv("heartTest.csv", )
    heartTestFeatures = heartTest.copy()
    heartTestLabels = heartTestFeatures.pop("chd")
    heartTestFeatures.pop("row.names")

    print("--Process data--")
    inputs, heartPreprocessing, heartTrainFeaturesDict = processcsv(heartTrainFeatures, heartTrain, True)

    print("--Make model--")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(10)
    ])

    preprocessedInputs = heartPreprocessing(inputs)
    result = model(preprocessedInputs)
    model = tf.keras.Model(inputs, result)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    print("--Fit model--")
    model.fit(heartTrainFeaturesDict, heartTrainLabels, epochs=2000, verbose=2)

    print("--Process test data--")
    inputs, heartPreprocessing, heartTestFeaturesDict = processcsv(heartTestFeatures, heartTrain, False)

    print("--Evaluate model--")
    model_loss, model_acc = model.evaluate(heartTestFeaturesDict, heartTestLabels, verbose=2)
    print(f"Model Loss:    {model_loss:.2f}")
    print(f"Model Accuracy: {model_acc * 100:.1f}%")


def processcsv(featurecsv, csv, preprocess):
    from tensorflow.keras.layers.experimental import preprocessing

    inputs = {}
    for name, column in featurecsv.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    numericInputs = {name: input for name, input in inputs.items()
                     if input.dtype == tf.float32}

    x = layers.Concatenate()(list(numericInputs.values()))
    if preprocess:
        norm = preprocessing.Normalization()
        norm.adapt(np.array(csv[numericInputs.keys()]))
        allNumericInputs = norm(x)
        preprocessedInputs = [allNumericInputs]
    else:
        preprocessedInputs = [x]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        lookup = preprocessing.StringLookup(vocabulary=np.unique(featurecsv[name]))
        oneHot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

        x = lookup(input)
        x = oneHot(x)
        preprocessedInputs.append(x)

    preprocessedInputsCat = layers.Concatenate()(preprocessedInputs)
    preprocessing = tf.keras.Model(inputs, preprocessedInputsCat)

    featuresDict = {name: np.array(value)
                    for name, value in featurecsv.items()}

    return inputs, preprocessing, featuresDict


if __name__ == "__main__":
    main()
