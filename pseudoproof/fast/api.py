from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import io

# from pseudoproof.ml_logic.model import *
from pseudoproof.ml_logic.preproc import clean_data, scale_data, digit_freq
from pseudoproof.cloud.load_models import load_models, load_RFmodel
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


@app.post("/predict_RF")
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

    # call RF model
    model = await load_RFmodel()

    # prepare df to add prediction results
    prediction_df = X_clean.copy()

    # predict
    res_series = model.predict(X_final)

    # add to df
    prediction_df[f"prediction"] = res_series.astype(float)

    # create percent of fake
    percent = np.mean(res_series) * 100
    pred_percent = round(percent, 1)

    # df including all predictions, dictionary with percentage of fabricated rows per model
    return prediction_df, pred_percent


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

    # df including all predictions, dictionary with percentage of fabricated rows per model
    return prediction_df, pred_percent
