from flask import request
from flask_socketio import join_room, leave_room, emit

import re
import time
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
                game_state.assign_player_sid(data["cachedPlayerId"], request.sid)
                emit_start_game(game_state, room_id)
                emit("assign_player", {
                    "playerId": data["cachedPlayerId"]
                })
                emit_start_turn(game_state, room_id)
                # TODO: what if we're resuming an ended game?

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
        canvas_image = Image.open(image_data_io)

        # save state
        game_state.canvasImage = canvas_image
        # TODO: save preview data

        if data["playerId"] != game_state.turn.id:
            print("WARNING: Received wrong player id for end turn")
            return

        game_state.next_turn()
        emit_start_turn(game_state, room_id)

        if game_state.can_end_game():
            emit_end_game(game_state, game_config, data["preview"], room_id)


    @socketio.on("disconnect")
    def disconnect():
        print(f"Disconnection: {request.sid}")
        found_player, game, room_id = games_manager.get_player_location_by_sid(request.sid)
        if found_player is None:
            print(f"WARNING: Could not find player with sid {request.sid}")
            return

        # check if player reconnects in time
        found_player.sid = None
        time.sleep(Settings.get("PAGE_REFRESH_BUFFER_TIME"))
        if found_player.sid is not None:
            return

        # remove player
        found_player.sid = request.sid   # accounts for players with duplicate sids
        game.remove_player(request.sid)  # as is the case with local play

        # TODO: check for game ending
        #if game.can_end_game():
        #    emit_end_game(game_state, game_config, data["preview"], room_id)

        # check for room removal and service stoppage
        if len(game.players) <= 0:
            label_pair = game.label_pair
            games_manager.remove_room(room_id)

            games_manager.label_pair_rooms[label_pair].remove(room_id)

            if len(games_manager.label_pair_rooms[label_pair]) <= 0:
                games_manager.stop_model_service(label_pair)
                del games_manager.label_pair_rooms[label_pair]


def emit_start_turn(game_state, room_id):
    emit("start_turn", {
        "canvas": game_state.canvasImageToSerial(),
        "turn": game_state.turn.id,
        "target": game_state.turn.target,
        "turnsLeft": game_state.turns_left,
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
        },
        "totalNumTurns": game_state.total_num_turns,
    }, to=room_id)


def emit_end_game(game_state, game_config, image_data, room_id):
    model_service_base = Settings.get("MODEL_SERVICE_BASE", "http://localhost:5002")

    response = requests.post(
        f"{model_service_base}/infer",
        headers={ "Content-Type": "application/json" },
        data=json.dumps({
            "gameConfig": game_config,
            "label_pair": game_state.label_pair,
            "imageDataUrl": image_data
        })
    )

    winner_index = int(response.json()["modelOutputs"][1] > response.json()["modelOutputs"][0])
    winner_target = game_state.label_pair[winner_index]

    emit("end_game", {
        "winnerTarget": winner_target,
    }, to=room_id)
