from flask import Blueprint, render_template, redirect, request, Response

from .game import GameType
from .GameManager import GameManager
from .utils import GAME_CONFIG
from .sockets import emit_ai_stroke


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


    """ Receive AI stroke """


    @routes.route("/ai_stroke", methods=["POST"])
    def ai_stroke():
        # TODO: assert it's coming from model service
        emit_ai_stroke(request.json["strokeSamples"], request.json["roomId"])
        
        return "", 200


    return routes
