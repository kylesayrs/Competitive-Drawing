import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
import boto3
import requests
from flask_socketio import join_room, leave_room, emit

from utils.game import GameType


def make_socket_messages(socketio, game_config, games_manager):
    @socketio.on("join_room")
    def on_join_room(data):
        print(f"on_join_room: {data}")
        room_id = data.get("room_id")
        join_room(room_id)

        # Create new game if necessary
        if room_id not in games_manager.rooms:
            # TODO: throw some sort of error
            print(f"ERROR: unknown room id {room_id} not found in {games_manager.rooms}")
            return

        print(games_manager.rooms)
        game_state = games_manager.rooms[room_id]
        print(game_state.started)

        if not game_state.started:
            # add players
            if game_state.game_type == GameType.FREE_PLAY:
                player_one = game_state.add_player()
                player_two = game_state.add_player()
                emit("assign_player", {
                    "playerId": player_one.id
                })
                emit("assign_player", {
                    "playerId": player_two.id
                })

            if game_state.game_type == GameType.LOCAL:
                player_one = game_state.add_player()
                player_two = game_state.add_player()
                emit("assign_player", {
                    "playerId": player_one.id
                })
                emit("assign_player", {
                    "playerId": player_two.id
                })

            if game_state.game_type == GameType.ONLINE:
                if game_state.can_add_player():
                    new_player = game_state.add_player()

                    emit("assign_player", {
                        "playerId": new_player.id
                    })

            # attempt to start game
            if game_state.can_start_game:
                #  TODO: start_game -> game_state that includes canvas state
                #  send no matter what to account for refreshing
                emit_start_game(game_state, room_id)
                game_state.started = True
                emit_start_turn(game_state, room_id)
                start_model_service(game_state.label_pair)

        else:
            # resume game
            emit_start_game(game_state, room_id)
            emit_start_turn(game_state, room_id)

    @socketio.on("end_turn")
    def end_turn(data):
        print("end_turn")
        print(data)
        room_id = data["roomId"]
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

        else:
            print("WARNING: Received wrong player id for end turn")


def emit_start_turn(game_state, room_id):
    emit("start_turn", {
        "canvas": game_state.canvasImageToSerial(),
        "turn": game_state.turn.id
    }, to=room_id)


def emit_start_game(game_state, room_id):
    onnx_url = game_state.get_onnx_url()

    emit("start_game", {
        "onnxUrl": onnx_url,
        "canvas": game_state.canvasImageToSerial(),
        "targets": {
            player.id: player.target
            for player in game_state.players
        }
    }, to=room_id)


def start_model_service(label_pair):
    model_service_base = os.environ.get("MODEL_SERVICE_BASE", "http://localhost:5002")
    url = f"{model_service_base}/start_model"

    requests.post(
        url,
        headers={ "Content-Type": "application/json" },
        data=json.dumps({ "label_pair": label_pair }),
    )


def stop_model_service(label_pair, game_config):
    model_service_base = os.environ.get("MODEL_SERVICE_BASE", "http://localhost:5002")
    url = f"{model_service_base}/start_model"

    requests.post(
        url,
        headers={ "Content-Type": "application/json" },
        body=json.dumps(game_config),
    )
