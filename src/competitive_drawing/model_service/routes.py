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

        inferencer = model_manager.get_inferencer(request.json["label_pair"])

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


    """
    @routes.route("/infer_stroke", methods=["POST"])
    def infer_stroke():
        # TODO: move this to utils file
        image_data_url = request.json["imageDataUrl"]
        image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
        image_data = base64.b64decode(image_data_str)
        image_data_io = BytesIO(image_data)
        base_canvas = Image.open(image_data_io)

        target_index = request.json["targetIndex"]
        max_length = request.json["maxLength"]

        inferencer = model_manager.get_inferencer(request.json["label_pair"])
        score_model = inferencer.model

        # TODO: Cheat detection

        loss, keypoints = grid_search_stroke(
            base_canvas,
            (3, 3),
            score_model,
            request.json["targetIndex"],
            torch.optim.Adamax,
            { "lr": 0.02 },
            max_width=5.0,
            min_width=1.5,
            max_aa=0.35,
            min_aa=0.9,
            max_steps=100,
            save_best=True,
            draw_output=True,
            max_length=max_length,
        )

        curve = BezierCurve(
            keypoints,
            sample_method="uniform",
            num_approximations=20
        )
        curve_points = [curve.sample(t) for t in get_uniform_ts(20)]

        return json.dumps({"curve_points": curve_points}), 200
    """


    @routes.route("/stop_model", methods=["POST"])
    def stop_model():
        try:
            model_manager.stop_model(request.json["label_pair"])
            return "", 200

        except Exception as exception:
            print(exception)
            return str(exception), 409

    return routes
