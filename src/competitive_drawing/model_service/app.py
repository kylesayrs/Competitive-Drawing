# general
import os
import json

# flask
from flask import Flask
from flask_cors import CORS

# implementations
from .routes import make_routes_blueprint
from .utils import ModelManager
from competitive_drawing import Settings


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, resources={ r"/*": {"origins": "*"} }) #Settings.get("ALLOWED_ORIGIN", "localhost:5001")
    app.config["SECRET_KEY"] = Settings.get("MODEL_SERVICE_SECRET_KEY", "secret!")

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    # model manager
    model_manager = ModelManager()

    # routes
    routes_blueprint = make_routes_blueprint(model_manager)
    app.register_blueprint(routes_blueprint)

    return app


def start_app():
    app = create_app()
    app.run(
        host=Settings.get("MODEL_SERVICE_HOST", "localhost"),
        port=Settings.get("MODEL_SERVICE_PORT", 5002),
        debug=True
    )


if __name__ == "__main__":
    start_app()
