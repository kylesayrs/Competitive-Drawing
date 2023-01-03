import boto3
from flask_socketio import join_room, leave_room, emit

from utils.game import GameType, GameState


def make_socket_messages(socketio, games_manager):
    @socketio.on("join_room")
    def on_join_room(data):
        print(f"on_join_room: {data}")
        room_id = int(data.get("room_id"))
        join_room(room_id)

        # Create new game if necessary
        if room_id not in games_manager.rooms:
            games_manager.rooms[room_id] = GameState(data.get("game_type"))

        game_state = games_manager.rooms[room_id]

        if game_state.game_type == GameType.FREE_PLAY.value:
            player_one = game_state.add_player()
            player_two = game_state.add_player()
            emit("assign_player", {
                "playerId": player_one.id
            })
            emit("assign_player", {
                "playerId": player_two.id
            })

        if game_state.game_type == GameType.LOCAL.value:
            player_one = game_state.add_player()
            player_two = game_state.add_player()
            emit("assign_player", {
                "playerId": player_one.id
            })
            emit("assign_player", {
                "playerId": player_two.id
            })
            print(player_two)
            print(game_state.players)

        if game_state.game_type == GameType.ONLINE.value:
            if game_state.can_add_player():
                new_player = game_state.add_player()

                emit("assign_player", {
                    "playerId": new_player.id
                })

        print(game_state)
        print(game_state.can_start_game)
        print(game_state.started)
        if game_state.can_start_game and not game_state.started:
            #  TODO: start_game -> game_state that includes canvas state
            #  send no matter what to account for refreshing
            emit_start_game(game_state, room_id)

            game_state.started = True
            emit_start_turn(game_state, room_id)


    @socketio.on("end_turn")
    def end_turn(data):
        print("end_turn")
        print(data)
        room_id = int(data["roomId"])
        game_state = games_manager.rooms[room_id]

        image_data_url = data["canvas"]
        image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
        image_data = base64.b64decode(image_data_str)
        image_data_io = BytesIO(image_data)
        image = Image.open(image_data_io)

        game_state.canvasImage = image

        if data["playerId"] == game_state.turn.id:
            game_state.next_turn()
            emit_start_turn(game_state, room_id)


def emit_start_turn(game_state, room_id):
    emit("start_turn", {
        "canvas": game_state.canvasImageToSerial(),
        "turn": game_state.turn.id
    }, to=room_id)


def emit_start_game(game_state, room_id):
    onnx_url = game_state.get_onnx_url()
    print(onnx_url)

    emit("start_game", {
        "onnxUrl": onnx_url,
        "canvas": game_state.canvasImageToSerial(),
        "targets": {
            player.id: player.target
            for player in game_state.players
        }
    }, to=room_id)