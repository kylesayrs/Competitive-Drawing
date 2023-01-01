# general
import os
import json
from dotenv import load_dotenv

# canvas to numpy (to be moved)
from io import BytesIO, StringIO
from PIL import Image
import re

# flask
from flask import Flask
from flask_socketio import SocketIO

# implementations
from routes import make_routes_blueprint
from utils.models import GameState, Player, GameManager

load_dotenv(".env")


def create_app():
    # get environment variables
    host = os.environ.get("HOST", "localhost")
    port = os.environ.get("PORT", 5000)
    api_root = f"http://{host}:{port}"
    secret_key = os.environ.get("SECRET_KEY", "secret!")
    model_checkpoint_path = os.environ.get("MODEL_PATH", "./static/models/model.pth")

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = secret_key
    socketio = SocketIO(app)

    # set up config to pass to javascript
    game_config = {
        "softmaxFactor": os.environ.get("SOFTMAX_FACTOR", 5),
        "canvasSize": os.environ.get("CANVAS_SIZE", 100),
        "distancePerTurn": os.environ.get("DISTANCE_PER_TURN", 80)
    }

    # set up games manager
    games_manager = GameManager()

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    # routes
    routes_blueprint = make_routes_blueprint(game_config, games_manager)
    app.register_blueprint(routes_blueprint)

    # socketio
    #games_manager = GameManager()
    #make_socket_messages(socketio, games_manager)

    return app, socketio

if __name__ == "__main__":
    host = os.environ.get("HOST", "localhost")
    port = os.environ.get("PORT", 5000)

    app, socketio = create_app()
    socketio.run(app, host=host, port=port, debug=True)
