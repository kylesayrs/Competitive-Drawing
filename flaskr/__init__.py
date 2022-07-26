import os
import json

from flask import Flask, request, render_template, redirect
from flask_socketio import SocketIO, join_room, leave_room, Namespace, emit

from .inference import infer_image

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    api_root = os.environ.get("API_ROOT", "http://localhost:5000") # TODO launch with this root too
    app.config['SECRET_KEY'] = 'secret!'
    socketio = SocketIO(app)
    socketio.run(app)

    # create instance folder
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        raise ValueError("Could not create instance folder")

    """
    class MyCustomNamespace(Namespace):
        def on_connect(self):
            pass

        def on_disconnect(self):
            pass

        def on_my_event(self, data):
            emit('my_response', data)

    socketio.on_namespace(MyCustomNamespace('/test'))
    """

    @app.route("/", methods=["GET"])
    def home():
        return redirect("/select", code=302)

    @app.route("/select", methods=["GET"])
    def select():
        return render_template("game_room.html")

    @app.route("/game_room", methods=["GET"])
    def game_room():
        return render_template("game_room.html")

    @app.route("/infer", methods=["POST"])
    def infer():
        print("TODO: infer image")
        image_data = request.json["imageData"]
        confidences = infer_image(image_data)
        # TODO cheat detection
        return app.response_class(
            response=json.dumps({
                "confidences": confidences,
                "isCheater": False,
            }),
            status=200,
            mimetype='application/json'
        )

    @socketio.on("join")
    def on_join(data):
        print("on_join")
        print(data)
        join_room(data["room"])
        print("asdf")
        send("someone has entered the room", to=room)

    @socketio.on('leave')
    def on_leave(data):
        username = data['username']
        room = data['room']
        leave_room(room)
        send(username + ' has left the room.', to=room)

    return app

if __name__ == "__main__":
    create_app()
