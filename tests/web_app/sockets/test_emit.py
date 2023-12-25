import pytest
from flask import request
from flask_socketio import SocketIOTestClient

from competitive_drawing.web_app.app import app, socketio
from competitive_drawing.web_app.sockets import emit_assign_player


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
    [
        ("player_id"),
        ("None"),
        (""),
    ],
)
def test_emit_assign_player(player_id):
    test_client = SocketIOTestClient(app, socketio)
    test_client.connect()
    assert test_client.is_connected()

    client_sid = get_client_sid(test_client)

    emit_assign_player(player_id, client_sid)

    received_data = test_client.get_received()

    found_data = [data for data in received_data if data["name"] == "assign_player"]
    assert len(found_data) == 1
    data_args = found_data[0]["args"][0]
    assert "playerId" in data_args 
    assert data_args["playerId"] == player_id
