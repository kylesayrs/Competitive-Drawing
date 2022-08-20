from typing import List, Dict

import uuid
import random
import numpy
from PIL import Image

class Player:
    id: str
    target: str

    def __init__(self, id, target):
        self.id = id
        self.target = target

class GameState:
    canvas: List[List[List[int]]]
    labels: List[str]
    players: List[Player]
    started: bool
    _player_turn_index: int

    def __init__(self, labels):
        self.canvasImage = Image.new("RGB", (500, 500), (255, 255, 255))
        self.labels = labels
        self.players = []
        self._player_turn_index = 0
        self.started = False

    def can_add_player(self):
        return len(self.players) < 2

    def add_player(self):
        target_index = random.randint(0, len(self.labels) - 1) #  TODO: avoid overlap
        new_player = Player(
            id=uuid.uuid4().hex,
            target=self.labels[target_index],
        )
        self.players.append(new_player)
        return new_player

    def can_start_game(self):
        return len(self.players) >= 2

    @property
    def turn(self):
        return self.players[self._player_turn_index]

    def next_turn(self):
        self._player_turn_index = (self._player_turn_index + 1) % len(self.players)

    def canvasImageToSerial(self):
        return numpy.array(self.canvasImage).tolist()

class GameManager:
    rooms: Dict[int, GameState]

    def __init__(self):
        self.rooms = {}
