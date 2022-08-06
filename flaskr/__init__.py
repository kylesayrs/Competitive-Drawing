import os
import json
import onnx
import base64
from io import BytesIO, StringIO
from PIL import Image
import re

from flask import Flask, request, render_template, redirect
from flask_socketio import SocketIO, join_room, leave_room, Namespace, emit, send

from .inference import Inferencer

ALL_LABELS = [
    "sheep",
    "guitar",
    "pig",
    "tree",
    "clock",
    "squirrel",
    "duck",
    "panda",
    "spider",
    "snowflake",
]

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    api_root = os.environ.get("API_ROOT", "http://localhost:5000") # TODO launch with this root too
    app.config['SECRET_KEY'] = 'secret!'
    socketio = SocketIO(app)
    socketio.run(app)

    # load model TODO move to inference.py
    inferencer = Inferencer(
        model_checkpoint_path=os.environ.get(
            "MODEL_PATH", "./flaskr/static/models/model.pth"
        )
    )

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    @app.route("/", methods=["GET"])
    def home():
        return redirect("/select", code=302)

    @app.route("/select", methods=["GET"])
    def select():
        return render_template("select.html")

    @app.route("/free_draw", methods=["GET"])
    def free_draw():
        return render_template("free_draw.html", data={"allLabels": ALL_LABELS})

    @app.route("/local_game", methods=["GET"])
    def local_game():
        return render_template("local_game.html", data={"allLabels": ALL_LABELS})

    @app.route("/game_room", methods=["GET"])
    def game_room():
        return render_template("game_room.html", data={"allLabels": ALL_LABELS})

    @app.route("/infer", methods=["POST"])
    def infer():
        image_data_url = request.json["imageDataUrl"]
        image_data_str = re.sub('^data:image/.+;base64,', '', image_data_url)
        image_data = base64.b64decode(image_data_str)
        image_data_io = BytesIO(image_data)
        image = Image.open(image_data_io)

        target_index = request.json["targetIndex"]

        #model_outputs = inferencer.infer_image(image)
        model_outputs, grad_cam_image = inferencer.infer_image_with_cam(image, target_index)

        # TODO grad cam
        # TODO cheat detection
        return app.response_class(
            response=json.dumps({
                "modelOutputs": model_outputs,
                "gradCamImage": grad_cam_image,
                "isCheater": False,
            }),
            status=200,
            mimetype='application/json'
        )

    @socketio.on("select")
    def select(payload):
        if payload.get("type") == "localgame":
            emit("goto", "/localgame")

    return app

if __name__ == "__main__":
    create_app()
