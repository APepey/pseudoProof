from google.cloud import storage
import joblib
from pseudoproof.params import *

client = storage.Client()


def download_models() -> None:
    """Downloads file from the Google Bucket"""

    BUCKET_NAME = "pseudoproof"
    bucket = client.bucket(BUCKET_NAME)
    all_files = bucket.list_blobs()
    pickle_names = [each.name for each in all_files if ".pkl" in each.name]
    for pickle_file_name in pickle_names:
        local_blob = bucket.blob(pickle_file_name)
        local_blob.download_to_filename(
            os.path.join(LOCAL_MODEL_PATH, f"{pickle_file_name}")
        )
    return pickle_names


def load_models():
    "Loads the model from the local pickle file"
    pickle_names = download_models()
    model_dict = {}

    for pickle_file_name in pickle_names:
        # my_list.append(joblib.load(os.path.join(LOCAL_MODEL_PATH,f'{pickle_file_name}')))
        model_dict[pickle_file_name] = joblib.load(
            os.path.join(LOCAL_MODEL_PATH, f"{pickle_file_name}")
        )

    return model_dict
