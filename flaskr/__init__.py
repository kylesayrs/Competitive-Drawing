import os

from flask import Flask, render_template


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    # a simple page that says hello
    @app.route('/')
    def home():
        image_shape = (28, 28)
        canvas_shape = tuple(size * 25 for size in image_shape)
        return render_template(
            "base.html",
            image_shape=image_shape,
            canvas_shape=canvas_shape
        )

    return app
