from flask import Flask, request
from markupsafe import escape
from gevent.pywsgi import WSGIServer
import json
from pathlib import Path
import os
import re
import tensorflow as tf
import numpy as np


print('Current server path:', Path().resolve())
models_folder = Path("./server_models")

app = Flask(__name__)

def load_model(name, version):
    path = models_folder / name / str(version)

    if os.path.exists(path):
        f = [p for p in os.listdir(path) if p.endswith(".keras")]
        if len(f) == 1:
            return tf.keras.models.load_model(path / f[0])


def load_best_model(model_name):
    versions = os.listdir(models_folder / model_name)

    rg = re.compile(r"\d+")
    versions = set([int(s) for s in versions if rg.match(s)])
    versions = list([s for s in versions if str(s) == str(int(s))])
    versions.sort(reverse=True)

    for ver in versions:
        try:
            model = load_model(model_name, ver)
            if not (model is None):
                return model
        except Exception as e:
            print(e)

    raise Exception("Model not found or can't load any version of model")


def best_model_predict(model_name, input_data):
    model = load_best_model(model_name)

    pred_data = model.predict(input_data)
    return pred_data


@app.route("/models/<model>/:predict", methods=["POST"])
def main(model):

    try:
        ## model ##

        data = json.loads(request.data)

        if not (model in os.listdir(models_folder)):
            return {"error": f"not found model: {model}"}

        ## version ##

        versions = os.listdir(models_folder / model)

        rg = re.compile(r"\d+")
        versions = set([int(s) for s in versions if rg.match(s)])
        versions = set([s for s in versions if str(s) == str(int(s))])

        if len(versions) != 0:
            version = str(max(versions))
        else:
            return {"error": f"no versions found for model \"{model}\""}

        # model = load_best_model(model)

        if not ('instances' in data):
            raise Exception('No field "instances" found in your data')
        
        data = np.array(data['instances'])

        pred = best_model_predict(
            model_name=model, 
            input_data=data
        )
        
        return {'prdictions': pred.tolist()}
    except Exception as e:
        print(e)
        return {'error': str(e)}


if __name__ == "__main__":
    # app.run(port=8000)
    http_server = WSGIServer(("", 8502), app)
    print("Server is running. Ctrl+C to interrupt.")
    http_server.serve_forever()
