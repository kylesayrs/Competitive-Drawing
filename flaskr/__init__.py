import os
import json

from flask import Flask, request, render_template

from .inference import infer_image

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    api_root = os.environ.get("API_ROOT", "http://localhost:5000") # TODO launch with this root too

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    # a simple page that says hello
    @app.route("/")
    def home():
        return render_template(
            "base.html",
            inference_url="/".join((api_root, "infer")),
        )

    @app.route("/infer", methods=["POST"])
    def infer():
        print("TODO: infer image")
        image_data = request.json["imageData"]
        confidences = infer_image(image_data)
        return app.response_class(
            response=json.dumps({"confidences": confidences}),
            status=200,
            mimetype='application/json'
        )

    return app
