# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
import base64
import io
import json
import os
from abc import ABC

import numpy as np
import torch
from captum.attr import IntegratedGradients
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from PIL import Image

from .base_handler import BaseHandler


class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """

    def __init__(self):
        super().__init__()

    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None
        if (
            "DALI_PREPROCESSING" in os.environ
            and os.environ["DALI_PREPROCESSING"].lower() == "true"
        ):

            self.model_dir = properties.get("model_dir")
            config_file = open(self.model_dir + "/dali_config.json")
            self.configs = json.load(config_file)

    def dali_preprocess(self, data):
        batch_tensor = []
        batch_size = self.configs["batch_size"]
        num_threads = self.configs["num_threads"]
        device_id = self.configs["device_id"]
        prefetch_queue_depth = self.configs["prefetch_queue_depth"]

        input_byte_arrays = [i["body"] if "body" in i else i["data"] for i in data]
        for byte_array in input_byte_arrays:
            np_image = np.frombuffer(byte_array, dtype=np.uint8)
            batch_tensor.append(np_image)  # we can use numpy

        filename = self.model_dir + "/model.dali"
        pipe = Pipeline.deserialize(filename=filename)
        pipe._max_batch_size = batch_size
        pipe._num_threads = num_threads
        pipe._device_id = device_id

        for _ in range(prefetch_queue_depth):
            pipe.feed_input("my_source", batch_tensor)

        datam = PyTorchIterator(
            [pipe],
            ["data"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_padded=True,
        )
        result = []
        for i, data in enumerate(datam):
            result.append(data[0]["data"])
            break

        return result[0].to(self.device)

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        if (
            "DALI_PREPROCESSING" in os.environ
            and os.environ["DALI_PREPROCESSING"].lower() == "true"
        ):
            return self.dali_preprocess(data=data)

        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
