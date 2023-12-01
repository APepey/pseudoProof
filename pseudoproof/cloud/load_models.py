from google.cloud import storage
import joblib
from pseudoproof.params import *

client = storage.Client()


def download_model() -> None:
    """Downloads file from the Google Bucket"""

    BUCKET_NAME = "pseudoproof"
    bucket = client.bucket(BUCKET_NAME)
    all_files = bucket.list_blobs()
    pickle_names = [each.name for each in all_files if ".pkl" in each.name]
    pickle_file_name = pickle_names[0]
    # blob = file
    local_blob = bucket.blob(pickle_file_name)
    local_blob.download_to_filename(os.path.join(LOCAL_MODEL_PATH, "saved_model.pkl"))


def load_model():
    "Loads the model from the local pickle file"
    download_model()
    model = joblib.load(os.path.join(LOCAL_MODEL_PATH, "saved_model.pkl"))
    return model
