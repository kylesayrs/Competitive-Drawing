from typing import List, Dict

import uuid
import random

class Player:
    id: str
    target: str

    def __init__(self, id, target):
        self.id = id
        self.target = target

class GameState:
    labels: List[str]
    players: List[Player]
    started: bool
    _player_turn_index: int

    def __init__(self, labels):
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

class GameManager:
    rooms: Dict[int, GameState]

    def __init__(self):
        self.rooms = {}
