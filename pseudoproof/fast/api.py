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
import asyncio

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


# app.state.model = asyncio.run(load_models())
# app.state.model = load_models()


@app.get("/status")
def index():
    return {"status": "ok"}


@app.post("/predict")
async def predict(csv: UploadFile = File(...)):
    # make the uploaded csv file usable
    bytes_object = await csv.read()
    byte_string = str(bytes_object, "utf-8")
    data = StringIO(byte_string)

    with open("input.csv", "w") as file:
        print(data.getvalue(), file=file)

    df = pd.read_csv("input.csv")

    # preprocess the data
    X_clean = clean_data(df)
    X_scaled = scale_data(X_clean)
    X_final = digit_freq(X_scaled)

    # call models
    model_dict = await load_models()
    model_list = list(model_dict.keys())

    # prepare df and dict to add prediction results
    prediction_df = X_clean.copy()
    pred_percent = {}

    # creating a table with a new column for corresponding row prediction
    for model_name in model_list:
        # extract model from model name
        clean_name = model_name.split(".")[0]
        # retrieve model
        model = model_dict[model_name]
        # create a series with all predictioms
        res_series = model.predict(X_final)
        # append to a column corresponding to model name and predictions
        prediction_df[f"prediction_{clean_name}"] = res_series.astype(float)
        # complete dict with percentage of fabricated rows
        percent = np.mean(res_series) * 100
        pred_percent[clean_name] = round(percent, 1)

    # create a different df for each model?

    # df including all predictions, dictionary with percentage of fabricated rows per model
    return prediction_df, pred_percent


## other functions not in use at the moment
@app.post("/predict_one_row")
async def predict(csv: UploadFile = File(...), n=0):
    bytes_oobject = await csv.read()
    byte_string = str(bytes_oobject, "utf-8")
    data = StringIO(byte_string)

    with open("input.csv", "w") as file:
        print(data.getvalue(), file=file)

    df = pd.read_csv("input.csv")

    X_clean = clean_data(df)
    X_scaled = scale_data(X_clean)
    X_final = digit_freq(X_scaled)

    model_dict = await load_models()  # app.state.model
    model_list = list(model_dict.keys())

    prediction = {}

    for model_name in model_list:
        clean_name = model_name.split(".")[0]

        model = model_dict[model_name]
        model_prediction = float(
            model.predict(X_final)[n]
        )  # choosing line to predict here
        prediction[clean_name] = model_prediction

    return prediction


@app.get("/NN_predict")
def NNmodel_predict():
    # preprocessing
    df = clean_data(contents)

    # separating X and y
    X = df.drop(columns=["y"])
    y = df[["y"]]

    # standard scaling
    X_scaled = scale_data(X)

    # split

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_split=0.3)

    # X_train, X_test = X_scaled[:-1], X_scaled[-1:]
    # y_train, y_test = y[:-1], y[-1:]

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
