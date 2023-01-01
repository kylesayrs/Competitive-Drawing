from flask import Blueprint, render_template, redirect


def make_routes_blueprint(game_config, games_manager):
    routes = Blueprint("routes", __name__)

    @routes.route("/", methods=["GET"])
    def home():
        return redirect("/select", code=302)

    @routes.route("/select", methods=["GET"])
    def select():
        return render_template("select.html")

    @routes.route("/free_draw", methods=["GET"])
    def free_draw():
        return render_template("free_draw.html", game_config=game_config)

    @routes.route("/local_game", methods=["GET"])
    def local_game():
        return render_template("local_game.html", game_config=game_config)

    @routes.route("/game_room", methods=["GET"])
    def game_room():
        return render_template("game_room.html", game_config=game_config)

    # jank
    @routes.route("/reset_rooms", methods=["GET"])
    def reset_rooms():
        games_manager.rooms = {}

        response = app.response_class(
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