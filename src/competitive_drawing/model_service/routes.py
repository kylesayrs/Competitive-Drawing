from flask import Blueprint, request

import json

from .utils import imageDataUrlToImage


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
        image = imageDataUrlToImage(request.json["imageDataUrl"])

        try:
            inferencer = model_manager.get_inferencer(request.json["label_pair"])
        except Exception as exception:
            print(exception)
            return str(exception), 409

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


    @routes.route("/infer_stroke", methods=["POST"])
    def infer_stroke():
        image = imageDataUrlToImage(request.json["imageDataUrl"])
        target_index = request.json["targetIndex"]
        line_width = (
            request.json["gameConfig"]["canvasLineWidth"] /
            request.json["gameConfig"]["canvasSize"] *
            request.json["gameConfig"]["imageSize"]
        )
        max_length = (
            request.json["gameConfig"]["distancePerTurn"] /
            request.json["gameConfig"]["canvasSize"] *
            request.json["gameConfig"]["imageSize"]
        )

        # TODO: Cheat detection

        inferencer = model_manager.get_inferencer(request.json["label_pair"])
        stroke_samples = inferencer.infer_stroke(image, target_index, line_width, max_length)

        return json.dumps({"strokeSamples": stroke_samples}), 200


    @routes.route("/stop_model", methods=["POST"])
    def stop_model():
        try:
            model_manager.stop_model(request.json["label_pair"])
            return "", 200

        except Exception as exception:
            print(exception)
            return str(exception), 409

    return routes
