from flask import request
from flask_socketio import join_room, leave_room, emit

import re
import json
import base64
from io import BytesIO
from PIL import Image
import boto3
import requests

from competitive_drawing.web_app.utils.game import GameType
from competitive_drawing import Settings


def make_socket_messages(socketio, game_config, games_manager):
    @socketio.on("join_room")
    def on_join_room(data):
        game_type = GameType(data["game_type"])
        room_id = data["room_id"]
        sid = request.sid

        # Create new game if necessary
        if room_id not in games_manager.rooms[game_type]:
            # TODO: throw some sort of error
            print(f"ERROR: unknown room id {room_id} not found in {games_manager.rooms}")
            return

        # join socket room
        join_room(room_id)

        game_state = games_manager.get_game(game_type, room_id)

        if not game_state.started:
            if game_state.game_type == GameType.FREE_PLAY:
                player_one = game_state.add_player(sid)
                player_two = game_state.add_player(sid)
                emit("assign_player", {
                    "playerId": player_one.id
                })
                emit("assign_player", {
                    "playerId": player_two.id
                })

            elif game_state.game_type == GameType.LOCAL:
                player_one = game_state.add_player(sid)
                player_two = game_state.add_player(sid)
                emit("assign_player", {
                    "playerId": player_one.id
                })
                emit("assign_player", {
                    "playerId": player_two.id
                })

            elif game_state.game_type == GameType.ONLINE:
                if game_state.can_add_player():
                    new_player = game_state.add_player(sid)
                    emit("assign_player", {
                        "playerId": new_player.id
                    })

            elif game_state.game_type == GameType.SINGLE_PLAYER:
                player_one = game_state.add_player(sid)
                player_two = game_state.add_player(sid)

            else:
                print(f"WARNING: Unknown game type {game_state.game_type}")

            # start game
            if game_state.can_start_game:
                game_state.start_game()
                emit_start_game(game_state, room_id)
                emit_start_turn(game_state, room_id)

        else:
            # resume game if player id is valid
            if game_state.has_player(data["cachedPlayerId"]):
                emit_start_game(game_state, room_id)
                emit("assign_player", {
                    "playerId": data["cachedPlayerId"]
                })
                emit_start_turn(game_state, room_id)

            else:
                print(
                    f"WARNING: Unknown player with id {data['cachedPlayerId']} tried "
                    f"to join game with players {game_state.players}"
                )

    @socketio.on("end_turn")
    def end_turn(data):
        game_type = GameType(data["game_type"])
        room_id = data["roomId"]
        game_state = games_manager.get_game(game_type, room_id)

        # TODO: break out into utils.helpers
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


    @socketio.on("disconnect")
    def disconnect():
        print(request.sid)
        games_manager.remove_player_from_game_room(request.sid)


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
        },
        "targetIndices": {
            player.id: player.target_index
            for player in game_state.players
        }
    }, to=room_id)
