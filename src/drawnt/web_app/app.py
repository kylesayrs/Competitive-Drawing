# general
import os
import json

# flask
from flask import Flask
from flask_socketio import SocketIO

# implementations
from .routes import make_routes_blueprint
from .sockets import make_socket_messages
from .utils.game import GameManager
from drawnt import Settings


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = Settings.get("WEB_APP_SECRET_KEY", "secret!")
    socketio = SocketIO(app)

    # set up config to pass to javascript
    game_config = {
        "softmaxFactor": float(Settings.get("SOFTMAX_FACTOR", 5)),
        "canvasSize": int(Settings.get("CANVAS_SIZE", 100)),
        "canvasLineWidth": float(Settings.get("CANVS_LINE_WIDTH", 1.5)),
        "imageSize": int(Settings.get("IMAGE_SIZE", 50)),
        "imagePadding": int(Settings.get("IMAGE_PADDING", 0)),
        "distancePerTurn": float(Settings.get("DISTANCE_PER_TURN", 80)),
        "staticCrop": bool(Settings.get("STATIC_CROP", 1)),
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


def start_app():
    host = Settings.get("WEB_APP_HOST", "localhost")
    port = Settings.get("WEB_APP_PORT", 5001)

    app, socketio = create_app()
    socketio.run(app, host=host, port=port, debug=True)


if __name__ == "__main__":
    start_app()
