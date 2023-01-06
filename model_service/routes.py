from flask import Blueprint, request

import json
import base64
from io import BytesIO
from PIL import Image
import re

def make_routes_blueprint(app, model_manager):
    routes = Blueprint("routes", __name__)

    @routes.route("/start_model", methods=["POST"])
    def start_model():
        try:
            model_manager.start_model(request.json["label_pair"])
            return "", 200

        except Exception as exception:
            print(exception)
            return str(exception), 409


    @routes.route("/infer", methods=["POST"])
    def infer():
        # TODO: move this to utils file
        image_data_url = request.json["imageDataUrl"]
        image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
        image_data = base64.b64decode(image_data_str)
        image_data_io = BytesIO(image_data)
        image = Image.open(image_data_io)

        inferencer = model_manager.get_model(request.json["label_pair"])

        # TODO: Cheat detection

        if "targetIndex" in request.json:
            target_index = request.json["targetIndex"]
            model_outputs, grad_cam_image = inferencer.infer_image_with_cam(image, target_index)

            response_data = {
                "modelOutputs": model_outputs,
                "gradCamImage": grad_cam_image,
                "isCheater": False,
            }

        else:
            model_outputs = inferencer.infer_image(image)

            response_data = {
                "modelOutputs": model_outputs,
                "isCheater": False,
            }

        return json.dumps(response_data), 200


    @routes.route("/stop_model", methods=["POST"])
    def stop_model():
        try:
            model_manager.stop_model(request.json["label_pair"])
            return "", 200

        except Exception as exception:
            print(exception)
            return str(exception), 409

    return routes
