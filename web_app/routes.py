import os
from flask import Blueprint, render_template, redirect, request

from utils.game import GameType


def make_routes_blueprint(app, game_config, games_manager):
    routes = Blueprint("routes", __name__)

    @routes.route("/", methods=["GET"])
    def home():
        return redirect("/select")

    @routes.route("/select", methods=["GET"])
    def select():
        return render_template("select.html")

    @routes.route("/free_draw", methods=["GET"])
    def free_draw():
        return render_template("free_draw.html", game_config=game_config)

    @routes.route("/local_game", methods=["GET"])
    def local_game():
        room_id = request.args.get("room_id")
        if room_id is None:
            room_id = games_manager.assign_game_room(GameType.LOCAL)
            return redirect(f"local_game?room_id={room_id}")
        else:
            return render_template("local_game.html", game_config=game_config)

    @routes.route("/game_room", methods=["GET"])
    def game_room():
        room_id = request.args.get("room_id")
        if room_id is None:
            room_id = games_manager.assign_game_room(GameType.ONLINE)
            return redirect(f"game_room?room_id={room_id}")
        else:
            return render_template("game_room.html", game_config=game_config)

    @routes.route("/infer", methods=["POST"])
    def infer():
        model_service_base = os.environ.get("MODEL_SERVICE_BASE", "http://localhost:5002")
        return redirect(f"{model_service_base}/infer", code=307)

    # jank
    @routes.route("/reset_rooms", methods=["GET"])
    def reset_rooms():
        games_manager.rooms = {}

        response = app.Response(
            response=json.dumps({
                "status": "success",
                "code": 0,
                "rooms": games_manager.rooms,
            }),
            status=200,
            mimetype="application/json"
        )
        return response

    return routes
