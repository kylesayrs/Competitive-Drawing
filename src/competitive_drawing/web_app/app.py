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
from competitive_drawing import Settings

SETTINGS = Settings()


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = SETTINGS.web_app_secret_key
    socketio = SocketIO(app)

    # set up config to pass to javascript
    game_config = {
        "softmaxFactor": SETTINGS.softmax_factor,
        "canvasSize": SETTINGS.canvas_size,
        "canvasLineWidth": SETTINGS.canvas_line_width,
        "imageSize": SETTINGS.image_size,
        "imagePadding": SETTINGS.image_padding,
        "distancePerTurn": SETTINGS.distance_per_turn,
        "staticCrop": SETTINGS.static_crop
    }

    # set up games manager
    games_manager = GameManager()

    # create instance folder
    os.makedirs(app.instance_path, exist_ok=True)

    # routes
    routes_blueprint = make_routes_blueprint(app, game_config, games_manager)
    app.register_blueprint(routes_blueprint)

    # socketio
    make_socket_messages(socketio, game_config, games_manager)

    return app, socketio


def start_app():
    app, socketio = create_app()
    socketio.run(
        app,
        host=SETTINGS.web_app_host,
        port=SETTINGS.web_app_port,
        debug=True
    )


if __name__ == "__main__":
    start_app()
