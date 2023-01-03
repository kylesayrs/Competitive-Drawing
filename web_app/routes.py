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

    @routes.route("/infer", methods=["POST"])
    def infer():
        # TODO: move this to utils file
        image_data_url = request.json["imageDataUrl"]
        image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
        image_data = base64.b64decode(image_data_str)
        image_data_io = BytesIO(image_data)
        image = Image.open(image_data_io)

        # TODO cheat detection

        if "targetIndex" in request.json:
            target_index = request.json["targetIndex"]
            model_outputs, grad_cam_image = inferencer.infer_image_with_cam(image, target_index)

            return route.response_class(
                response=json.dumps({
                    "modelOutputs": model_outputs,
                    "gradCamImage": grad_cam_image,
                    "isCheater": False,
                }),
                status=200,
                mimetype="application/json"
            )

        else:
            model_outputs = inferencer.infer_image(image)

            return route.response_class(
                response=json.dumps({
                    "modelOutputs": model_outputs,
                    "isCheater": False,
                }),
                status=200,
                mimetype="application/json"
            )

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