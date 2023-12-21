from typing import Dict, Any

import requests
from flask import Blueprint, render_template, redirect, request, Response
from flask_socketio import emit

from competitive_drawing import Settings
from .game import GameType
from .GameManager import GameManager
from .utils import GAME_CONFIG

SETTINGS = Settings()


def make_routes_blueprint(games_manager: GameManager) -> Blueprint:
    routes = Blueprint("routes", __name__)

    """ General """


    @routes.route("/", methods=["GET"])
    def home():
        return redirect("/tutorial")

    @routes.route("/select", methods=["GET"])
    def select():
        return render_template("select.html")

    @routes.route("/tutorial", methods=["GET"])
    def tutorial():
        return render_template("tutorial.html")

    @routes.route("/free_draw", methods=["GET"])
    def free_draw():
        return render_template("free_draw.html", game_config=GAME_CONFIG)
    

    """ Games """


    @routes.route("/local", methods=["GET"])
    def local_game():
        if request.args.get("room_id") is None:
            room_id = games_manager.assign_game_room(GameType.LOCAL)
            return redirect(f"local?room_id={room_id}")

        return render_template("local.html", game_config=GAME_CONFIG)

    @routes.route("/online", methods=["GET"])
    def online():
        if request.args.get("room_id") is None:
            room_id = games_manager.assign_game_room(GameType.ONLINE)
            return redirect(f"online?room_id={room_id}")
        
        return render_template("online.html", game_config=GAME_CONFIG)

    @routes.route("/single_player", methods=["GET"])
    def single_player():
        if request.args.get("room_id") is None:
            room_id = games_manager.assign_game_room(GameType.SINGLE_PLAYER)
            return redirect(f"single_player?room_id={room_id}")

        return render_template("single_player.html", game_config=GAME_CONFIG)


    """ Inference """


    @routes.route("/infer", methods=["POST"])
    def infer():
        service_response = requests.post(
            f"{SETTINGS.model_service_base}/infer",
            headers={"Content-Type": "application/json"},
            data=request.get_data(),
        )

        excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
        service_headers = [
            (k,v) for k,v in service_response.raw.headers.items()
            if k.lower() not in excluded_headers
        ]

        return Response(service_response.content, service_response.status_code, service_headers)

    @routes.route("/infer_stroke", methods=["POST"])
    def infer_stroke():
        service_response = requests.post(
            f"{SETTINGS.model_service_base}/infer_stroke",
            headers={"Content-Type": "application/json"},
            data=request.get_data(),
        )

        excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
        service_headers = [
            (k,v) for k,v in service_response.raw.headers.items()
            if k.lower() not in excluded_headers
        ]

        return Response(service_response.content, service_response.status_code, service_headers)

    @routes.route("/ai_stroke", methods=["POST"])
    def ai_stroke():
        # TODO: assert it's coming from model service
        emit("ai_stroke", {
            "strokeSamples": request.json["strokeSamples"]
        }, namespace="/", to=request.json["roomId"])
        return "", 200


    return routes
