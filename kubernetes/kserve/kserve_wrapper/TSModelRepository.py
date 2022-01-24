""" The repository to serve the Torchserve Models in the kserve side"""
import logging
import sys
import json
import kserve
from tornado.httpclient import AsyncHTTPClient
from kserve.model_repository import ModelRepository

TS_MODEL_STATUS_FORMAT = "http://{0}/models/{1}"

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

class TSModelRepository(ModelRepository):
    """A repository of kserve KFModels
    Args:
        KFModelRepository (object): The parameters from the KFModelRepository is passed
        as inputs to the TSModel Repository.
    """
    def __init__(self, inference_address: str, management_address: str,
                 model_dir: str):
        """The Inference Address, Management Address and the Model Directory from the kserve
        side is initialized here.

        Args:
            inference_address (str): The Inference Address present in the kserve side.
            management_address (str): The Management Address present in the kserve side.
            model_dir (str): the directory of the model artefacts in the kserve side.
        """
        super().__init__(model_dir)
        logging.info("TSModelRepo is initialized")
        self.inference_address = inference_address
        self.management_address = management_address
        self.model_dir = model_dir

    async def is_model_ready(self, name: str) -> bool:
        headers = {"Content-Type": "application/json; charset=UTF-8"}
        response = await AsyncHTTPClient(max_clients=sys.maxsize).fetch(
            TS_MODEL_STATUS_FORMAT.format(self.inference_address.split("//")[1], name),
            method="GET",
            headers=headers,
        )
        response_body = json.loads(response.body)
        if response_body[0].get('workers')[0].get('status') == 'READY':
            self.ready = True
        return self.ready
