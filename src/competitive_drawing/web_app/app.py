# general
import os
import json

# flask
from flask import Flask
from flask_socketio import SocketIO

# implementations
from .routes import make_routes_blueprint
from .sockets import make_socket_callbacks
from .GameManager import GameManager
from competitive_drawing import Settings

SETTINGS = Settings()


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = SETTINGS.web_app_secret_key
    socketio = SocketIO(app)

    # set up games manager
    games_manager = GameManager()

    # create instance folder
    os.makedirs(app.instance_path, exist_ok=True)

    # routes
    routes_blueprint = make_routes_blueprint(games_manager)
    app.register_blueprint(routes_blueprint)

    # socketio
    make_socket_callbacks(socketio, games_manager)

    return app, socketio


# export
app, socketio = create_app()


def start_app():
    socketio.run(
        app,
        host=SETTINGS.web_app_host,
        port=SETTINGS.web_app_port,
        debug=True
    )


if __name__ == "__main__":
    start_app()
