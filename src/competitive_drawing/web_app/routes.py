import os
from flask import Blueprint, render_template, redirect, request
from flask_socketio import emit

from .utils.game import GameType
from competitive_drawing import Settings


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

    @routes.route("/local", methods=["GET"])
    def local_game():
        room_id = request.args.get("room_id")
        if room_id is None:
            room_id = games_manager.assign_game_room(GameType.LOCAL)
            return redirect(f"local?room_id={room_id}")
        else:
            return render_template("local.html", game_config=game_config)

    @routes.route("/online", methods=["GET"])
    def online():
        room_id = request.args.get("room_id")
        if room_id is None:
            room_id = games_manager.assign_game_room(GameType.ONLINE)
            return redirect(f"online?room_id={room_id}")
        else:
            return render_template("online.html", game_config=game_config)

    @routes.route("/single_player", methods=["GET"])
    def single_player():
        room_id = request.args.get("room_id")
        if room_id is None:
            room_id = games_manager.assign_game_room(GameType.SINGLE_PLAYER)
            return redirect(f"single_player?room_id={room_id}")
        else:
            return render_template("single_player.html", game_config=game_config)


    @routes.route("/infer", methods=["POST"])
    def infer():
        model_service_base = Settings.get("MODEL_SERVICE_BASE", "http://localhost:5002")
        return redirect(f"{model_service_base}/infer", code=307)

    @routes.route("/infer_stroke", methods=["POST"])
    def infer_stroke():
        model_service_base = Settings.get("MODEL_SERVICE_BASE", "http://localhost:5002")
        return redirect(f"{model_service_base}/infer_stroke", code=307)


    @routes.route("/ai_stroke", methods=["POST"])
    def ai_stroke():
        # TODO: assert it's coming from model service
        emit("ai_stroke", {
            "strokeSamples": request.json["strokeSamples"]
        }, namespace="/", to=request.json["roomId"])
        return "", 200


    return routes
