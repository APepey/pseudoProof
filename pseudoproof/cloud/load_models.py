from google.cloud import storage
import joblib
from pseudoproof.params import *
from prefect_gcp import GcpCredentials
import os


async def get_credentials_from_prefect():
    gcp_credentials_block = await GcpCredentials.load("first-block")
    credentials = gcp_credentials_block.get_credentials_from_service_account()
    return credentials


async def download_models() -> None:
    """Downloads file from the Google Bucket"""

    credentials = await get_credentials_from_prefect()
    client = storage.Client(credentials=credentials)

    BUCKET_NAME = "pseudoproof"
    bucket = client.bucket(BUCKET_NAME)
    all_files = bucket.list_blobs()
    pickle_names = [each.name for each in all_files if ".pkl" in each.name]
    for pickle_file_name in pickle_names:
        local_blob = bucket.blob(pickle_file_name)
        local_blob.download_to_filename(
            os.path.join(LOCAL_MODEL_PATH, f"{pickle_file_name}")
        )

    print(pickle_names)
    return pickle_names


# async def load_model():
#     model = await load_models()
#     return model


# @app.on_event("startup")
# async def startup_event():
#     app.state.model = await load_models()


async def load_RFmodel():
    """Load the RF model from the local pickle file"""
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    RFmodel = joblib.load(os.path.join(LOCAL_MODEL_PATH, "random_forest.pkl"))

    return RFmodel


async def load_models():
    """Loads the model from the local pickle file"""
    pickle_names = await download_models()
    model_dict = {}

    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    for pickle_file_name in pickle_names:
        model_dict[pickle_file_name] = joblib.load(
            os.path.join(LOCAL_MODEL_PATH, f"{pickle_file_name}")
        )

    return model_dict
