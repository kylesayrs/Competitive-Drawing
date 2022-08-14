# general
import os
import json
from dotenv import load_dotenv

# canvas to numpy (to be moved)
import base64
from io import BytesIO, StringIO
from PIL import Image
import re

# flask
from flask import Flask, request, render_template, redirect
from flask_socketio import SocketIO, join_room, leave_room, Namespace, emit, send

# implementations
from .inference import Inferencer
from .models import GameState, Player, GameManager

# TODO: move to .env
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
    # get dot env
    load_dotenv(".env")
    host = os.environ.get("HOST", "localhost")
    root = os.environ.get("ROOT", "5000")
    api_root = os.environ.get("API_ROOT", "http://localhost:5000")
    secret_key = os.environ.get("SECRET_KEY", "secret!")

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config['SECRET_KEY'] = secret_key
    socketio = SocketIO(app)
    socketio.run(app, host=host, root=root)

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
        # TODO: move this to utils file
        image_data_url = request.json["imageDataUrl"]
        image_data_str = re.sub('^data:image/.+;base64,', '', image_data_url)
        image_data = base64.b64decode(image_data_str)
        image_data_io = BytesIO(image_data)
        image = Image.open(image_data_io)

        target_index = request.json["targetIndex"]

        #model_outputs = inferencer.infer_image(image)
        model_outputs, grad_cam_image = inferencer.infer_image_with_cam(image, target_index)

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

    game_manager = GameManager()

    # jank
    @app.route("/reset_rooms", methods=["GET"])
    def reset_rooms():
        game_manager.rooms = {}

        response = app.response_class(
            response=json.dumps({
                "status": "success",
                "code": 0,
                "rooms": game_manager.rooms,
            }),
            status=200,
            mimetype='application/json'
        )
        return response

    @socketio.on("join_room")
    def on_join_room(data):
        room_id = data.get("room_id")
        room_id = int(room_id)
        join_room(room_id)

        if room_id not in game_manager.rooms:
            game_manager.rooms[room_id] = GameState(ALL_LABELS)

        game_state = game_manager.rooms[room_id]
        if game_state.can_add_player():
            new_player = game_state.add_player()

            emit("assign_player", {"playerId": new_player.id})

            if game_state.can_start_game() and not game_state.started:
                start_game_data = {
                    "targets": {
                        player.id: player.target
                        for player in game_state.players
                    },
                    "turn": game_state.players[0].id
                }
                emit("start_game", start_game_data, to=room_id)
                game_state.started = True

    return app

if __name__ == "__main__":
    create_app()
