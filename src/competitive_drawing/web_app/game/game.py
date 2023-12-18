from typing import List, Optional, Tuple, Union

import uuid
import random
import numpy
from PIL import Image

from competitive_drawing import Settings
from ..game import GameType, Player
from ..utils import get_available_label_pairs, get_onnx_url, label_pair_to_str

SETTINGS = Settings()


class Game:
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
        game_type: GameType,
        label_pair: Optional[Tuple[str, str]] = None,
        room_id: Optional[str] = None
    ):
        # game environment
        self.game_type = game_type
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
    

    @property
    def label_pair_str(self):
        return label_pair_to_str(self.label_pair)
    

    def add_player(self, sid: Union[str, None]):
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


    def canvas_image_to_serial(self) -> List[List[int]]:
        return numpy.array(self.canvas_image).tolist()


    def has_player(self, player_id: Union[str, None]):
        player_ids = [player.id for player in self.players]
        return player_id is not None and player_id in player_ids


    def start_game(self):
        self.started = True


    def remove_players_by_sid(self, sid: str):
        self.players = [
            player
            for player in self.players
            if player.sid != sid
        ]

    def can_end_game(self):
        return self.turns_left <= 0# or self.players <= 1


    def reassign_player_sid(self, player_id: str, new_sid: str):
        found_players = [player for player in self.players if player.id == player_id]
        if len(found_players) != 1:
            # TODO
            raise ValueError()
        
        found_player = found_players[0]

        found_player.sid = new_sid

    def get_other_player(self, player: Player) -> Player:
        found_players = [p for p in self.players if p.id != player.id]
        if len(found_players) != 1:
            # TODO
            raise ValueError()
    
        return found_players[1]


def _new_canvas_image() -> Image:
    canvas_shape = (SETTINGS.canvas_size, SETTINGS.canvas_size)
    return Image.new("RGB", canvas_shape, (255, 255, 255))


def _assign_label_pair() -> Tuple[str, str]:
    available_label_pairs = get_available_label_pairs()
    random.shuffle(available_label_pairs)
    return available_label_pairs[0]
