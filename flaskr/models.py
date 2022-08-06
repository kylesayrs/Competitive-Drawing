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
    player_turn_index: int
    started: bool

    def __init__(self, labels):
        self.labels = labels
        self.players = []
        self.player_turn_index = 0
        self.started = False

    def can_add_player(self):
        return len(self.players) < 2

    def add_player(self):
        target_index = random.randint(0, len(self.labels) - 1)
        new_player = Player(
            id=uuid.uuid4().hex,
            target=self.labels[target_index],
        )
        self.players.append(new_player)
        return new_player

    def can_start_game(self):
        return len(self.players) >= 2

class GameManager:
    rooms: Dict[int, GameState]

    def __init__(self):
        self.rooms = {}
