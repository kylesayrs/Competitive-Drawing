from typing import List, Optional, Tuple, Union
from abc import ABC, abstractclassmethod

import uuid
import random
import numpy
from PIL import Image

from competitive_drawing import SETTINGS
from ..game import GameType, Player
from ..utils import get_available_label_pairs, get_onnx_url, label_pair_to_str, data_url_to_image
from ..sockets import emit_start_game, emit_start_turn, emit_assign_player, emit_end_game
from ..model_service import server_infer


class Game(ABC):
    # game environment
    game_type: GameType
    label_pair: Tuple[str, str]
    room_id: str

    # game state
    canvas_image: Image  # used for resuming after a disconnect
    players: List[Player]
    started: bool
    _player_turn_index: int
    total_num_turns: int
    turns_left: int
    model_outputs: Tuple[float, float]

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
        self.model_outputs = [0.0, 0.0]  # fake score to make the game seem even at the beginning


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


    def next_turn(self, canvas_data_url: str, preview_data_url: str):
        self.canvas_image = data_url_to_image(canvas_data_url)
        self._player_turn_index = (self._player_turn_index + 1) % len(self.players)
        self.turns_left -= 1

        self.model_outputs = server_infer(self.room_id, self.label_pair, preview_data_url)

        if not self.can_end_game:
            emit_start_turn(self)        


    def canvas_image_to_serial(self) -> List[List[int]]:
        return numpy.array(self.canvas_image).tolist()


    def has_player(self, player_id: Union[str, None]) -> bool:
        player_ids = [player.id for player in self.players]
        return player_id is not None and player_id in player_ids


    def start_game(self):
        self.started = True

        emit_start_game(self)
        emit_start_turn(self)


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

        emit_start_game(self, new_sid)
        emit_assign_player(found_player.id, new_sid)
        emit_start_turn(self)
    

    def end_game(self, preview_data_url: str, force_loser: Player):
        if force_loser is not None:
            winner = self._get_other_player(force_loser)
            emit_end_game(self, winner.target)
            
        else:
            # do another inference for redundancy
            self.model_outputs = server_infer(self.room_id, self.label_pair, preview_data_url)

            # emit winner
            winner_index = numpy.argmax(self.model_outputs)
            winner_target = self.label_pair[winner_index]
            emit_end_game(self, winner_target)


    def _get_other_player(self, player: Player) -> Player:
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
