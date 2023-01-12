from typing import List, Dict, Optional, Tuple
from enum import Enum

import uuid
import random
import numpy
from PIL import Image

from .s3 import get_uploaded_label_pairs, get_onnx_url
from competitive_drawing import Settings


class GameType(Enum):
    FREE_PLAY = 0
    LOCAL = 1
    ONLINE = 2
    SINGLE_PLAYER = 3


class Player:
    id: str
    target: str

    def __init__(self, id, target):
        self.id = id
        self.target = target


class Game:
    game_type: GameType
    canvas: List[List[List[int]]]
    players: List[Player]
    started: bool
    _player_turn_index: int
    label_pair: Tuple[str, str]

    def __init__(
        self,
        game_type: GameType,
        label_pair: Optional[Tuple[str, str]] = None,
    ):
        self.game_type = game_type
        self.canvasSize = int(Settings.get("CANVAS_SIZE", 100))
        self.canvasImage = Image.new("RGB", (self.canvasSize, self.canvasSize), (255, 255, 255))
        self.players = []
        self._player_turn_index = 0
        self.started = False

        if label_pair is None:
            label_pairs = get_uploaded_label_pairs()
            random.shuffle(label_pairs)
            self.label_pair = label_pairs[0]
        else:
            self.label_pair = label_pair


    def can_add_player(self):
        return len(self.players) < 2


    def add_player(self):
        target_index = len(self.players)
        new_player = Player(
            id=uuid.uuid4().hex,
            target=self.label_pair[target_index],
        )
        self.players.append(new_player)
        return new_player


    @property
    def can_start_game(self):
        return (
            self.game_type == GameType.FREE_PLAY or
            len(self.players) >= 2
        )


    @property
    def turn(self):
        return self.players[self._player_turn_index]


    def next_turn(self):
        self._player_turn_index = (self._player_turn_index + 1) % len(self.players)


    def canvasImageToSerial(self):
        return numpy.array(self.canvasImage).tolist()


    def get_onnx_url(self):
        return get_onnx_url(self.label_pair)


class GameManager:
    rooms: Dict[GameType, Dict[str, Game]]

    def __init__(self):
        self.rooms = {
            game_type: {}
            for game_type in GameType
        }


    def assign_game_room(self, game_type: GameType, *game_args, **game_kwargs):
        game_rooms = self.rooms[game_type]
        available_game_room_ids = [
            room_id
            for room_id, game in game_rooms.items()
            if game.can_add_player()
        ]

        if len(available_game_room_ids) > 0:
            random.shuffle(available_game_room_ids)
            return available_game_room_ids[0]

        else:
            return self.new_game_room(game_type, *game_args, **game_kwargs)


    def new_game_room(self, game_type: GameType, *game_args, **game_kwargs):
        new_room_id = uuid.uuid4().hex
        new_game = Game(game_type, *game_args, **game_kwargs)

        self.rooms[game_type][new_room_id] = new_game
        return new_room_id


    def get_game(self, game_type: GameType, room_id: str):
        return self.rooms[game_type][room_id]
