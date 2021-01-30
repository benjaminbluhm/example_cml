import boto3
from pickle import load
from dvc import api
from urllib.parse import urlparse


class PythonPredictor:
    def __init__(self, config):
        s3 = boto3.client("s3")
        remote_path = api.get_url(config["model_path"], repo=config["dvc_repo"])
        o = urlparse(remote_path)
        bucket, key = o.netloc, o.path.lstrip('/')
        s3.download_file(bucket, key, "/tmp/trained_model.pkl")
        self.model = load(open("/tmp/trained_model.pkl", "rb"))

    def predict(self, payload):
        measurements = [
            float(payload["1"]), float(payload["2"]), float(payload["3"]), float(payload["4"]),
            float(payload["5"]), float(payload["6"]), float(payload["7"]), float(payload["8"]),
            float(payload["9"]),
        ]

        return str(self.model.predict([measurements])[0])


