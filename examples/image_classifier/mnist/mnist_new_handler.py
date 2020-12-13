import json
import logging
import os
import io
from captum.attr import IntegratedGradients
import torch
from PIL import Image
from ts.torch_handler.image_classifier import ImageClassifier

logger = logging.getLogger(__name__)


class MNISTDigitHandler(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """

    def __init__(self):
        super(MNISTDigitHandler, self).__init__()
        self.mapping_file_path = None

    def initialize(self, ctx):
        """
        First try to load torchscript else load eager mode state_dict based model
        :param ctx: System properties
        """
        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )
        print("the device name is",self.device)
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
       #model_pt_path = os.path.join(model_dir, "model.pth")
        from mnist_gpu import Net

        state_dict = torch.load("mnist_gpu.pt")
        self.model = Net()
        self.model.load_state_dict(state_dict)
        self.ig = IntegratedGradients(self.model)
        self.model.to(self.device)
        self.model.eval()

        #logger.debug("Model file %s loaded successfully", model_pt_path)

        #self.mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        self.initialized = True
    
    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
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

    
    def inference(self, img):
        """
        Predict the class (or classes) of an image using a trained deep learning model
        :param img: Input to be passed through the layers for prediction
        :return: output - Predicted label for the given input
        """
        # Convert 2D image to 1D vector
        # img = np.expand_dims(img, 0)
        # img = torch.from_numpy(img)
        self.model.eval()
        inputs = img.to(self.device)
        outputs = self.model.forward(inputs)

        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return [predicted_idx]

    def postprocess(self, inference_output):
        """
        Does postprocess after inference to be returned to user

        :param inference_output: Output of inference

        :return: output - Output after post processing
        """

        if self.mapping_file_path:
            with open(self.mapping_file_path) as json_file:
                data = json.load(json_file)
            inference_output = [json.dumps(data[inference_output[0]])]
            return inference_output
        return [inference_output]

     
    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
        
