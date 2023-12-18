from typing import Dict, Any, TYPE_CHECKING

from flask import request
from flask_socketio import join_room, leave_room

import time

from competitive_drawing import Settings
if TYPE_CHECKING:
    from ..game import GameManager


def make_socket_callbacks(socketio, game_manager: "GameManager"):
    @socketio.on("join_room")
    def on_join_room(data: Dict[str, Any]):
        room_id = data["roomId"]
        player_id = data["playerId"]
        sid = request.sid

        # join socket room
        join_room(room_id)

        # join game associated with room
        game_manager.join_game(room_id, sid, player_id)


    @socketio.on("end_turn")
    def end_turn(data: Dict[str, Any]):
        room_id = data["roomId"]
        player_id = data["playerId"]
        canvas_data_url = data["canvas"]
        canvas_preview_data_url = data["preview"]

        # may end game if criteria are met
        game_manager.end_turn(room_id, player_id, canvas_data_url, canvas_preview_data_url)


    @socketio.on("disconnect")
    def disconnect():
        print(f"Disconnection: {request.sid}")

        player, game = game_manager.player_game_by_sid[request.sid]
        if player is None:
            print(f"WARNING: Could not find player with sid {request.sid}")
            return

        # check if player reconnects in time
        # Note: there's a race condition if the client reconnects faster than it
        # takes the code to get to here. In this case, the client will be met with
        # a message asking to refresh, at which point it can try again
        player.sid = None
        time.sleep(Settings().client_disconnect_grace_period)
        if player.sid is not None:
            return

        # end game
        game_manager.end_game(game, force_loser=player)
