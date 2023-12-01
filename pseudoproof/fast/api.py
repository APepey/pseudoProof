from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import io
from pseudoproof.ml_logic.model import *
from pseudoproof.ml_logic.preproc import clean_data, scale_data
from pseudoproof.cloud.load_models import load_model

# creating decorator
app = FastAPI()

# Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()


@app.get("/status")
def index():
    return {"status": "ok"}


# adapt contents to dynamically retrieve data
contents = pd.read_csv("./raw_data/datasets/complete_dataset_true_fake.csv")


@app.post("/predict")
def predict(csv: UploadFile = File(...)):
    df = clean_data(contents)
    X = df.drop(columns=["y"])
    y = df[["y"]]
    X_scaled = scale_data(X)

    model = app.state.model
    prediction = model.predict(X_scaled)[0]
    return {"prediction": float(prediction)}


@app.get("/neural_predict")
def NNmodel_predict():
    # preprocessing
    df = clean_data(contents)

    # separating X and y
    X = df.drop(columns=["y"])
    y = df[["y"]]

    # standard scaling
    X_scaled = scale_data(X)

    # split

    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_split=0.3)

    X_train, X_test = X_scaled[:-1], X_scaled[-1:]
    y_train, y_test = y[:-1], y[-1:]

    layer_shape = X_train.shape[1:]

    # prediction RNN
    model = initialize_NNmodel(layer_shape)
    compile_NNmodel(model)
    trained_model, history = train_NNmodel(model, X_train, y_train)

    # evaluate_RNNmodel(model, X_test, y_test)
    y_pred = trained_model.predict(X_test)[0]

    if y_pred > 0.5:
        y_pred = 1
    else:
        y_pred = 0

    if y_pred == int(y_test["y"]):
        return {"result": "Banger"}


@app.get("/predict")
def predict():
    X_train, X_test, y_train, y_test = preproc(contents)
    layer_shape = X_train.shape[1:]
    model = initialize_NNmodel(layer_shape)
    compile_NNmodel(model)
    trained_model, history = train_NNmodel(model, X_train, y_train)
    y_pred = trained_model.predict(X_test)[0]

    return y_pred
