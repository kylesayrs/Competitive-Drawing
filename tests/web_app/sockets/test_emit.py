import pytest
import json
from flask import request
from flask_socketio import SocketIOTestClient

from competitive_drawing.web_app.app import app, socketio
from competitive_drawing.web_app.game import create_game, GameType, Player
from competitive_drawing.web_app.sockets import (
    emit_assign_player,
    emit_start_turn
)


def get_client_sid(client: SocketIOTestClient) -> str:
    global client_sid
    @socketio.on("connect")
    def connect():
        global client_sid
        client_sid = request.sid

    client.emit("connect")

    return client_sid


@pytest.mark.parametrize(
    "player_id",
    ["player_id", "None", ""],
)
def test_emit_assign_player(player_id):
    test_client = SocketIOTestClient(app, socketio)
    test_client.connect()
    assert test_client.is_connected()

    client_sid = get_client_sid(test_client)
    emit_assign_player(player_id, client_sid)

    assert (
        json.dumps(test_client.get_received()) ==
        f'[{{"name": "assign_player", "args": [{{"playerId": "{player_id}"}}], "namespace": "/"}}]'
    )


@pytest.mark.parametrize(
    "game_type",
    [GameType.LOCAL, GameType.ONLINE, GameType.SINGLE_PLAYER],
)
def test_emit_start_turn(game_type):
    game = create_game(game_type)
    two_player_game = game_type in [GameType.ONLINE]

    # reduce canvas image to make message checking easier
    game.canvas_image = []

    # client_a connect
    client_a = SocketIOTestClient(app, socketio)
    client_a.connect()
    assert client_a.is_connected()

    # client_b connect
    client_b = SocketIOTestClient(app, socketio)
    client_b.connect()
    assert client_b.is_connected()

    # client_a add (directly to avoid side effect emissions)
    client_a_sid = get_client_sid(client_a)
    player_a = Player(
        id="player_a",
        sid=client_a_sid,
        target=game.label_pair[0],
        target_index=0,
    )
    game.players.append(player_a)
    client_a.emit("join_room", {"roomId": game.room_id, "playerId": player_a.id})

    # client_b add (directly to avoid side effect emissions)
    if two_player_game:
        client_b_sid = get_client_sid(client_b)
        player_b = Player(
            id="player_b",
            sid=client_b_sid,
            target=game.label_pair[1],
            target_index=1,
        )
        game.players.append(player_b)
        client_b.emit("join_room", {"roomId": game.room_id, "playerId": player_b.id})

    # emit start turn
    emit_start_turn(game)

    # check received
    assert json.dumps(client_a.get_received()) == (
        '[{"name": "start_turn", "args": [{"canvas": [], "turn": "player_a", '
        f'"target": "{game.label_pair[0]}", "turnsLeft": 10, "modelOutputs": '
        '[0.0, 0.0]}], "namespace": "/"}]'
    )

    # check received
    if two_player_game:
        assert json.dumps(client_b.get_received()) == (
            '[{"name": "start_turn", "args": [{"canvas": [], "turn": "player_a", '
            f'"target": "{game.label_pair[0]}", "turnsLeft": 10, "modelOutputs": '
            '[0.0, 0.0]}], "namespace": "/"}]'
        )
