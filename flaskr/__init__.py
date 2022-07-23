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
        return render_template("base.html")
        #return 'Hello, World!'

    return app
