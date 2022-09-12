# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC
import io
import os
import base64
import torch
import numpy as np
from PIL import Image
from captum.attr import IntegratedGradients
from .base_handler import BaseHandler
from torchvision import transforms
from PIL import Image
from nvidia import dali
from nvidia.dali import types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

from nvidia.dali.plugin.pytorch import LastBatchPolicy

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
        if "DALI_PREPROCESSING" in os.environ and os.environ["DALI_PREPROCESSING"].lower() == "true":
            self.batch_tensor = []

    @pipeline_def(batch_size=5, num_threads=1, device_id=0)
    def dali_pipeline(self):
        jpegs = dali.fn.external_source(source=[self.batch_tensor], dtype=types.UINT8)
        jpegs = dali.fn.decoders.image(jpegs)
        resized = dali.fn.resize(jpegs, size=[256])
        normalized = dali.fn.crop_mirror_normalize(
            resized,
            crop_pos_x=0.5,
            crop_pos_y=0.5,
            crop=(224,224),
            mean=[0.485*255, 0.456*255, 0.406*255],
            std=[0.229*255, 0.224*255, 0.225*255])
        return normalized

    def dali_preprocess(self, data):
        input_byte_arrays = [i['body'] if 'body' in i else i['data'] for i in data]
        for byte_array in input_byte_arrays:
            np_image = np.frombuffer(byte_array, dtype = np.uint8)
            self.batch_tensor.append(np_image)  # we can use numpy
        # pii = PyTorchIterator(pipelines=[self.pipe], output_map=['data'])
        # for i, data in enumerate(pii):
        #    print("iter {}, real batch size: {}".format(i, len(data[0]["data"])))
        # pii.reset()
        # pipe_out, =self.pipe.run()
        result = []
        datam = PyTorchIterator([self.dali_pipeline()], ['data'], last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True)
        for i, data in enumerate(datam):
            result.append(data[0]['data'])
        self.batch_tensor = []

        # return torch.tensor(result).unsqueeze(0)
        return result[0].to(self.device)

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        if "DALI_PREPROCESSING" in os.environ and os.environ["DALI_PREPROCESSING"].lower() == "true":
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

