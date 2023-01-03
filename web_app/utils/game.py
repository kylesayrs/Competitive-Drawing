from typing import List, Dict, Optional, Tuple
from enum import Enum

import os
import uuid
import random
import numpy
from PIL import Image
from dotenv import load_dotenv

from utils.s3 import get_uploaded_label_pairs, get_onnx_url

load_dotenv(".env")

class GameType(Enum):
    FREE_PLAY = 0
    LOCAL = 1
    ONLINE = 2


class Player:
    id: str
    target: str

    def __init__(self, id, target):
        self.id = id
        self.target = target


class Game:
    canvas: List[List[List[int]]]
    players: List[Player]
    started: bool
    _player_turn_index: int

    def __init__(
        self,
        game_type: Optional[GameType] = GameType.FREE_PLAY,
        label_pair: Optional[Tuple[str, str]] = None,
    ):
        self.canvasSize = int(os.environ.get("CANVAS_SIZE", 50))
        self.canvasImage = Image.new("RGB", (self.canvasSize, self.canvasSize), (255, 255, 255))
        self.players = []
        self._player_turn_index = 0
        self.started = False
        self.game_type = game_type

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
        print(self.game_type)
        return (
            self.game_type == GameType.FREE_PLAY.value or
            len(self.players) >= 2
        )


    @property
    def turn(self):
        print(self.players)
        print(self._player_turn_index)
        return self.players[self._player_turn_index]


    def next_turn(self):
        self._player_turn_index = (self._player_turn_index + 1) % len(self.players)


    def canvasImageToSerial(self):
        return numpy.array(self.canvasImage).tolist()


    def get_onnx_url(self):
        return get_onnx_url(self.label_pair)


class GameManager:
    rooms: Dict[str, Game]

    def __init__(self):
        self.rooms = {}

    def new_game_room(self, *game_args, **game_kwargs):
        new_room_id = uuid.uuid4().hex
        new_game = Game(*game_args, **game_kwargs)

        self.rooms[new_room_id] = new_game
        return new_room_id


