from flask import Blueprint, request, copy_current_request_context

import json
import requests
from threading import Thread

from .manager import ModelManager
from .utils import imageDataUrlToImage

from competitive_drawing import Settings

SETTINGS = Settings()


def make_routes_blueprint(model_manager: ModelManager):
    routes = Blueprint("routes", __name__)

    @routes.route("/update", methods=["POST"])
    def update():
        label_pair_games = request.json["label_pair_games"]  # maps label pairs to number of games

        model_manager.scale(label_pair_games)
        print(label_pair_games)
        print(model_manager.inferencers.keys())

        return "", 200


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
        """
        Asyncronously begin thread to infer AI stroke. Stroke information is sent
        by a subsequent http request to the web app
        """
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
        room_id = request.json["roomId"]

        # TODO: Cheat detection

        @copy_current_request_context
        def _send_stroke(model_manager: ModelManager, room_id, *args):
            inferencer = model_manager.get_inferencer(request.json["label_pair"])
            stroke_samples = inferencer.infer_stroke(*args)

            requests.post(
                f"http://{SETTINGS.web_app_host}:{SETTINGS.web_app_port}/ai_stroke",
                headers={"Content-type": "application/json"},
                data=json.dumps({
                    "strokeSamples": stroke_samples,
                    "roomId": room_id,
                })
            )

        thread = Thread(
            target=_send_stroke,
            args=(model_manager, room_id, image, target_index, line_width, max_length)
        )
        thread.start()

        return "", 201
    

    return routes
