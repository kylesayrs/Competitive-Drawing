from typing import List, Optional, Tuple, Union
from abc import ABC, abstractclassmethod

import uuid
import random
import numpy
from PIL import Image

from competitive_drawing import Settings
from ..game import GameType, Player
from ..utils import get_available_label_pairs, get_onnx_url, label_pair_to_str
from ..sockets import emit_start_game, emit_start_turn, emit_assign_player

SETTINGS = Settings()


class Game(ABC):
    # game environment
    game_type: GameType
    label_pair: Tuple[str, str]
    room_id: str

    # game state
    canvas_image: Image
    players: List[Player]
    started: bool
    _player_turn_index: int
    total_num_turns: int
    turns_left: int

    def __init__(
        self,
        label_pair: Optional[Tuple[str, str]] = None,
        room_id: Optional[str] = None
    ):
        # game environment
        self.label_pair = label_pair if label_pair is not None else _assign_label_pair()
        self.room_id = room_id if room_id is not None else uuid.uuid4().hex

        # game state
        self.canvas_image = _new_canvas_image()
        self.players = []
        self._player_turn_index = 0
        self.started = False
        self.total_num_turns = SETTINGS.total_num_turns
        self.turns_left = self.total_num_turns


    @property
    def turn(self) -> Player:
        return self.players[self._player_turn_index]
    

    @property
    def onnx_url(self) -> str:
        return get_onnx_url(self.label_pair)
    

    @property
    def label_pair_str(self) -> str:
        return label_pair_to_str(self.label_pair)
    

    @property
    def can_start_game(self) -> bool:
        return self.game_type == GameType.FREE_PLAY or len(self.players) >= 2
    

    @property
    def can_end_game(self) -> bool:
        return self.turns_left <= 0
    

    def add_player(self, sid: Union[str, None]) -> Player:
        target_index = len(self.players)
        new_player = Player(
            id=uuid.uuid4().hex,
            sid=sid,
            target=self.label_pair[target_index],
            target_index=target_index,
        )
        self.players.append(new_player)

        return new_player


    def next_turn(self, canvas_image: numpy.ndarray):
        self.canvas_image = canvas_image
        self._player_turn_index = (self._player_turn_index + 1) % len(self.players)
        self.turns_left -= 1

        if not self.can_end_game:
            emit_start_turn(self, self.room_id)


    def canvas_image_to_serial(self) -> List[List[int]]:
        return numpy.array(self.canvas_image).tolist()


    def has_player(self, player_id: Union[str, None]) -> bool:
        player_ids = [player.id for player in self.players]
        return player_id is not None and player_id in player_ids


    def start_game(self):
        self.started = True

        emit_start_game(self, self.room_id)
        emit_start_turn(self, self.room_id)


    def remove_players_by_sid(self, sid: str):
        self.players = [
            player
            for player in self.players
            if player.sid != sid
        ]


    def reassign_player_sid(self, player_id: str, new_sid: str):
        found_players = [player for player in self.players if player.id == player_id]
        if len(found_players) != 1:
            raise ValueError()  # TODO
        
        found_player = found_players[0]
        found_player.sid = new_sid

        emit_start_game(self, self.room_id)
        emit_assign_player(found_player.id, new_sid)
        emit_start_turn(self, self.room_id)


    def get_other_player(self, player: Player) -> Player:
        """
        Used for determining winner when player leaves

        :param player: player not being searched
        :return: other player in game
        """
        found_players = [p for p in self.players if p.id != player.id]
        if len(found_players) != 1:
            raise ValueError()  # TODO
    
        return found_players[0]


def _new_canvas_image() -> Image:
    canvas_shape = (SETTINGS.canvas_size, SETTINGS.canvas_size)
    return Image.new("RGB", canvas_shape, (255, 255, 255))


def _assign_label_pair() -> Tuple[str, str]:
    available_label_pairs = get_available_label_pairs()
    random.shuffle(available_label_pairs)
    return available_label_pairs[0]
