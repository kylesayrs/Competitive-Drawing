# general
import os
import json
from dotenv import load_dotenv

# flask
from flask import Flask
from flask_cors import CORS

# implementations
from routes import make_routes_blueprint
from utils import ModelManager

load_dotenv(".env")


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, resources={ r"/*": {"origins": "*"} }) #os.environ.get("ALLOWED_ORIGIN", "localhost:5001")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret!")

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    # model manager
    model_manager = ModelManager()

    # routes
    routes_blueprint = make_routes_blueprint(app, model_manager)
    app.register_blueprint(routes_blueprint)

    return app

if __name__ == "__main__":
    host = os.environ.get("HOST", "localhost")
    port = os.environ.get("PORT", 5002)

    app = create_app()
    app.run(host=host, port=port, debug=True)
