# general
import os
import json

# flask
from flask import Flask
from flask_cors import CORS

# implementations
from .routes import make_routes_blueprint
from .manager import ModelManager
from competitive_drawing import Settings

SETTINGS = Settings()


def create_app() -> Flask:
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = SETTINGS.model_service_secret_key

    # allow web app origin requests
    web_app_origin = f"{SETTINGS.web_app_host}:{SETTINGS.web_app_port}"
    CORS(app, resources={ r"/*": {"origins": web_app_origin} })

    # create instance folder
    os.makedirs(app.instance_path, exist_ok=True)

    # model manager
    model_manager = ModelManager()

    # routes
    routes_blueprint = make_routes_blueprint(model_manager)
    app.register_blueprint(routes_blueprint)

    return app


def start_app():
    app = create_app()
    app.run(
        host=SETTINGS.model_service_host,
        port=SETTINGS.model_service_port,
        debug=True
    )


if __name__ == "__main__":
    start_app()
