from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import io
from pseudoproof.ml_logic.model import *
from pseudoproof.ml_logic.preproc import clean_data, scale_data, digit_freq
from pseudoproof.cloud.load_models import load_models
import csv

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

app.state.model = load_models()


@app.get("/status")
def index():
    return {"status": "ok"}


# adapt contents to dynamically retrieve data
contents = pd.read_csv("./raw_data/datasets/complete_dataset_true_fake.csv")


# working! modif preproc to have always 20 cols and train on that. complete models dict
@app.post("/predict")
async def predict(csv: UploadFile = File(...)):
    bytes_oobject = await csv.read()
    byte_string = str(bytes_oobject, "utf-8")
    data = StringIO(byte_string)

    with open("input.csv", "w") as file:
        print(data.getvalue(), file=file)

    df = pd.read_csv("input.csv")

    X_clean = clean_data(df)
    X_scaled = scale_data(X_clean)
    X_final = digit_freq(X_scaled)

    model_dict = app.state.model
    model_list = list(model_dict.keys())

    prediction = {}

    for model_name in model_list:
        clean_name = model_name.split(".")[0]

        model = model_dict[model_name]
        model_prediction = float(model.predict(X_final)[0])
        prediction[clean_name] = model_prediction

    return prediction


@app.get("/predict")
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
