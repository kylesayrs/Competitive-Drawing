from typing import List, Optional, Tuple, Union

import uuid
import random
import numpy
from PIL import Image

from competitive_drawing import Settings
from ..game import GameType, Player
from ..utils.s3 import get_uploaded_label_pairs, get_onnx_url


class Game:
    game_type: GameType
    canvas: List[List[List[int]]]
    players: List[Player]
    started: bool
    _player_turn_index: int
    label_pair: Tuple[str, str]
    total_num_turns: int
    turns_left: int

    def __init__(
        self,
        game_type: GameType,
        label_pair: Optional[Tuple[str, str]] = None,
    ):
        self.game_type = game_type
        self.canvasSize = Settings().canvas_size
        self.canvasImage = Image.new("RGB", (self.canvasSize, self.canvasSize), (255, 255, 255))
        self.players = []
        self._player_turn_index = 0
        self.started = False
        self.total_num_turns = Settings().total_num_turns
        self.turns_left = self.total_num_turns

        if label_pair is None:
            label_pairs = get_uploaded_label_pairs()
            random.shuffle(label_pairs)
            self.label_pair = tuple(label_pairs[0])
        else:
            self.label_pair = tuple(label_pair)


    @property
    def can_add_player(self):
        return len(self.players) < 2


    @property
    def can_start_game(self):
        return self.game_type == GameType.FREE_PLAY or len(self.players) >= 2


    @property
    def turn(self):
        return self.players[self._player_turn_index]
    

    @property
    def onnx_url(self):
        return get_onnx_url(self.label_pair)
    

    def add_player(self, sid):
        target_index = len(self.players)
        new_player = Player(
            id=uuid.uuid4().hex,
            sid=sid,
            target=self.label_pair[target_index],
            target_index=target_index,
        )
        self.players.append(new_player)
        return new_player


    def next_turn(self):
        self._player_turn_index = (self._player_turn_index + 1) % len(self.players)
        self.turns_left -= 1


    def canvasImageToSerial(self):
        return numpy.array(self.canvasImage).tolist()


    def has_player(self, player_id: Union[str, None]):
        player_ids = [player.id for player in self.players]
        return player_id is not None and player_id in player_ids


    def start_game(self):
        self.started = True


    def remove_player(self, sid):
        self.players = [
            player
            for player in self.players
            if player.sid != sid
        ]

    def can_end_game(self):
        return self.turns_left <= 0# or self.players <= 1


    def assign_player_sid(self, player_id, new_sid):
        # For single player games, the ai shares the same sid
        for player_index, player in enumerate(self.players):
            if player.id == player_id or self.game_type == GameType.SINGLE_PLAYER:
                player.sid = new_sid

    def get_player_by_sid(self, player_sid):
        for player_index, player in enumerate(self.players):
            if player.sid == player_sid:
                return player

        return None
