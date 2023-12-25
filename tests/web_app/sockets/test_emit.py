import pytest
import json
from flask import request
from flask_socketio import SocketIOTestClient, join_room

from competitive_drawing.web_app.app import app, socketio
from competitive_drawing.web_app.game import create_game, GameType, Player
from competitive_drawing.web_app.sockets import (
    emit_assign_player,
    emit_start_turn,
    emit_start_game
)


def get_client_sid(client: SocketIOTestClient) -> str:
    global client_sid
    @socketio.on("connect")
    def connect():
        global client_sid
        client_sid = request.sid

    client.emit("connect")

    return client_sid


def client_join_room(client: SocketIOTestClient, room_id: str):
    @socketio.on("test_join")
    def connect():
        join_room(room_id)

    client.emit("test_join")


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
    game.canvas_image = []  # don't check canvas image

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
    client_join_room(client_a, game.room_id)

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
        client_join_room(client_b, game.room_id)

    # emit start turn
    emit_start_turn(game)

    # check received
    assert json.dumps(client_a.get_received()) == (
        '[{"name": "start_turn", "args": [{"canvas": [], "turn": "player_a", '
        f'"target": "{player_a.target}", "turnsLeft": 10, "modelOutputs": '
        '[0.0, 0.0]}], "namespace": "/"}]'
    )

    # check received
    if two_player_game:
        assert json.dumps(client_b.get_received()) == (
            '[{"name": "start_turn", "args": [{"canvas": [], "turn": "player_a", '
            f'"target": "{player_a.target}", "turnsLeft": 10, "modelOutputs": '
            '[0.0, 0.0]}], "namespace": "/"}]'
        )


@pytest.mark.parametrize(
    "game_type,to_player_one",
    [
        (GameType.LOCAL, False),
        (GameType.ONLINE, False),
        (GameType.SINGLE_PLAYER, False),
        (GameType.LOCAL, True),
        (GameType.ONLINE, True),
        (GameType.SINGLE_PLAYER, True),
    ],
)
def test_emit_start_game(game_type, to_player_one):
    game = create_game(game_type)
    two_player_game = game_type in [GameType.ONLINE]
    game.canvas_image = []  # don't check canvas image

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
    client_join_room(client_a, game.room_id)

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
        client_join_room(client_b, game.room_id)

    # emit start turn
    emit_start_game(game, sid=client_a_sid if to_player_one else None)

    # check received
    if not two_player_game:
        client_a_data = client_a.get_received()
        del client_a_data[0]["args"][0]["onnxUrl"]  # don't check onnxUrl
        assert json.dumps(client_a_data) == (
            '[{"name": "start_game", "args": [{"canvas": [], "targets": {"player_a": '
            f'"{player_a.target}"}}, "targetIndices": {{"player_a": 0}}, '
            '"totalNumTurns": 10}], "namespace": "/"}]'
        )
    else:
        client_a_data = client_a.get_received()
        del client_a_data[0]["args"][0]["onnxUrl"]  # don't check onnxUrl
        assert json.dumps(client_a_data) == (
            '[{"name": "start_game", "args": [{"canvas": [], "targets": {"player_a": '
            f'"{player_a.target}", "player_b": "{player_b.target}"}}, '
            '"targetIndices": {"player_a": 0, "player_b": 1}, "totalNumTurns": 10}], '
            '"namespace": "/"}]'
        )
            
        if not to_player_one:
            client_b_data = client_b.get_received()
            del client_b_data[0]["args"][0]["onnxUrl"]  # don't check onnxUrl
            assert json.dumps(client_b_data) == (
                '[{"name": "start_game", "args": [{"canvas": [], "targets": {"player_a": '
                f'"{player_a.target}", "player_b": "{player_b.target}"}}, '
                '"targetIndices": {"player_a": 0, "player_b": 1}, "totalNumTurns": 10}], '
                '"namespace": "/"}]'
            )
