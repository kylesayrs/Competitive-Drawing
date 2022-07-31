import os
import json
import onnx
from onnx2pytorch import ConvertModel
import base64
from io import BytesIO, StringIO
from PIL import Image
import re

from flask import Flask, request, render_template, redirect
from flask_socketio import SocketIO, join_room, leave_room, Namespace, emit, send

from .inference import infer_image

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    api_root = os.environ.get("API_ROOT", "http://localhost:5000") # TODO launch with this root too
    app.config['SECRET_KEY'] = 'secret!'
    socketio = SocketIO(app)
    socketio.run(app)

    # load model
    pytorch_model = load_model(os.environ.get("MODEL_PATH", "./flaskr/static/models/model.onnx"))

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
        return render_template("free_draw.html")

    @app.route("/local_game", methods=["GET"])
    def local_game():
        return render_template("local_game.html")

    @app.route("/game_room", methods=["GET"])
    def game_room():
        return render_template("game_room.html")

    @app.route("/infer", methods=["POST"])
    def infer():
        print("TODO: infer image")
        canvas_url = request.json["canvasBlobUrl"]
        image_data = re.sub('^data:image/.+;base64,', '', canvas_url)
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        #image = Image.open(io.BytesIO(canvas_url))
        """
        #image_b64 = request.values['imageBase64']
        image_data = re.sub('^data:image/.+;base64,', '', canvas_url)#.decode('base64')
        print(image_data)
        image_PIL = Image.open(StringIO(image_data))
        print(image_PIL)
        """

        confidences = infer_image(image)

        # TODO grad cam
        # TODO cheat detection

        return app.response_class(
            response=json.dumps({
                "confidences": confidences,
                "gradCam": None,
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

def load_model(model_path):
    onnx_model = onnx.load(model_path)
    pytorch_model = ConvertModel(onnx_model)
    return pytorch_model

if __name__ == "__main__":
    create_app()
