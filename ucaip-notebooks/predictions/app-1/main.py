from fastapi import FastAPI, Request

import json
import numpy as np
import pickle
import os

from google.cloud import storage
from preprocess import MySimpleScaler
from sklearn.datasets import load_iris
import tensorflow as tf


app = FastAPI()
gcs_client = storage.Client()

with open("preprocessor.pkl", 'wb') as preprocessor_f, open("model.h5", 'wb') as model_f:
    gcs_client.download_blob_to_file(
        f"{os.environ['AIP_STORAGE_URI']}/preprocessor.pkl", preprocessor_f
    )
    gcs_client.download_blob_to_file(
        f"{os.environ['AIP_STORAGE_URI']}/model.h5", model_f
    )

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

_class_names = load_iris().target_names
_model = tf.keras.models.load_model("model.h5")
_preprocessor = preprocessor


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    """ health check to ensure HTTP server is ready to handle 
        prediction requests
    """
    return {"Status": "Running"}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    print(body)

    instances = body["instances"]
    inputs = np.asarray(instances)
    preprocessed_inputs = _preprocessor.preprocess(inputs)
    outputs = _model.predict(preprocessed_inputs)
    print(outputs)

    parameters = body.get("parameters", {})
    if parameters.get('probabilities'):
      return outputs.tolist()
    else:
      return {"predictions": [_class_names[class_num] for class_num 
                              in np.argmax(outputs, axis=1)]}
