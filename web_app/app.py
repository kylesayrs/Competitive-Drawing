# general
import os
import json
from dotenv import load_dotenv

# flask
from flask import Flask
from flask_socketio import SocketIO

# implementations
from routes import make_routes_blueprint
from sockets import make_socket_messages
from utils.game import GameManager

load_dotenv(".env")


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret!")
    socketio = SocketIO(app)

    # set up config to pass to javascript
    game_config = {
        "softmaxFactor": float(os.environ.get("SOFTMAX_FACTOR", 5)),
        "canvasSize": int(os.environ.get("CANVAS_SIZE", 100)),
        "canvasLineWidth": float(os.environ.get("CANVS_LINE_WIDTH", 1.5)),
        "imageSize": int(os.environ.get("IMAGE_SIZE", 50)),
        "imagePadding": int(os.environ.get("IMAGE_PADDING", 0)),
        "distancePerTurn": float(os.environ.get("DISTANCE_PER_TURN", 80)),
        "staticCrop": bool(os.environ.get("STATIC_CROP", 1)),
    }
    print(game_config)

    # set up games manager
    games_manager = GameManager()

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    # routes
    routes_blueprint = make_routes_blueprint(app, game_config, games_manager)
    app.register_blueprint(routes_blueprint)

    # socketio
    make_socket_messages(socketio, game_config, games_manager)

    return app, socketio

if __name__ == "__main__":
    host = os.environ.get("HOST", "localhost")
    port = os.environ.get("PORT", 5001)

    app, socketio = create_app()
    socketio.run(app, host=host, port=port, debug=True)
