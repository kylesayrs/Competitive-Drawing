from typing import TYPE_CHECKING, Optional, List
if TYPE_CHECKING:
    from ..game import Game

from flask_socketio import emit


def emit_assign_player(player_id: str, sid: str):
    from competitive_drawing.web_app import socketio  # avoid circular import

    socketio.emit("assign_player", {"playerId": player_id}, to=sid)


def emit_start_turn(game: "Game"):
    from competitive_drawing.web_app import socketio  # avoid circular import

    socketio.emit("start_turn", {
        "canvas": game.canvas_image_to_serial(),
        "turn": game.turn.id,
        "target": game.turn.target,
        "turnsLeft": game.turns_left,
        "modelOutputs": game.model_outputs
    }, to=game.room_id)


def emit_start_game(game: "Game", sid: Optional[str] = None):
    from competitive_drawing.web_app import socketio  # avoid circular import

    destination = sid if sid is not None else game.room_id
    socketio.emit("start_game", {
        "onnxUrl": game.onnx_url,
        "canvas": game.canvas_image_to_serial(),
        "targets": {
            player.id: player.target
            for player in game.players
        },
        "targetIndices": {
            player.id: player.target_index
            for player in game.players
        },
        "totalNumTurns": game.total_num_turns,
    }, to=destination)


def emit_end_game(game: "Game", winner_target: str):
    from competitive_drawing.web_app import socketio  # avoid circular import

    socketio.emit("end_game", {
        "winnerTarget": winner_target,
        "modelOutputs": game.model_outputs
    }, to=game.room_id)


def emit_ai_stroke(stroke_samples: List[List[float]], room_id: str):
    from competitive_drawing.web_app import socketio  # avoid circular import
    
    socketio.emit("ai_stroke", {
        "strokeSamples": stroke_samples
    }, to=room_id)
    
