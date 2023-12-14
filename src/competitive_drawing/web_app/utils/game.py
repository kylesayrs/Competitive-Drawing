from typing import List, Dict, Optional, Tuple, Union
from enum import Enum

import json
import uuid
import random
import numpy
import requests
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
    sid: str
    target: str
    target_index: int

    def __init__(self, id, sid, target, target_index):
        self.id = id
        self.sid = sid
        self.target = target
        self.target_index = target_index


    def __str__(self):
        return (
            "Player(id = {self.id}, sid = {self.sid}, "
            "target = {self.target}, target_index = {self.target_index})"
        )


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
        self.canvasSize = int(Settings.get("CANVAS_SIZE", 100))
        self.canvasImage = Image.new("RGB", (self.canvasSize, self.canvasSize), (255, 255, 255))
        self.players = []
        self._player_turn_index = 0
        self.started = False
        self.total_num_turns = int(Settings.get("TOTAL_NUM_TURNS"))
        self.turns_left = self.total_num_turns

        if label_pair is None:
            label_pairs = get_uploaded_label_pairs()
            random.shuffle(label_pairs)
            self.label_pair = tuple(label_pairs[0])
        else:
            self.label_pair = tuple(label_pair)


    def can_add_player(self):
        return len(self.players) < 2


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
        self.turns_left -= 1

    def canvasImageToSerial(self):
        return numpy.array(self.canvasImage).tolist()


    def get_onnx_url(self):
        return get_onnx_url(self.label_pair)


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


class GameManager:
    rooms: Dict[GameType, Dict[str, Game]]
    label_pair_rooms: Dict[Tuple[str, str], List[str]]

    def __init__(self):
        self.rooms = {
            game_type: {}
            for game_type in GameType
        }
        self.label_pair_rooms = {}


    def assign_game_room(self, game_type: GameType, *game_args, **game_kwargs):
        game_rooms = self.rooms[game_type]
        available_game_room_ids = [
            room_id
            for room_id, game in game_rooms.items()
            if not game.started and game.can_add_player()
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

        if new_game.label_pair not in self.label_pair_rooms.keys():
            self.start_model_service(new_game.label_pair, new_room_id)

        if new_game.label_pair not in self.label_pair_rooms:
            self.label_pair_rooms[new_game.label_pair] = []

        self.label_pair_rooms[new_game.label_pair].append(new_room_id)

        return new_room_id


    def get_game(self, game_type: GameType, room_id: str):
        return self.rooms[game_type][room_id]


    def has_label_pair(self, label_pair):
        for game_type_rooms in self.rooms.values():
            for room_id, game in game_type_rooms.items():
                if game.label_pair == label_pair:
                    return True

        return False


    def get_player_location_by_sid(self, sid):
        found_player = None
        for game_type_rooms in self.rooms.values():
            for room_id, game in game_type_rooms.items():
                found_player = game.get_player_by_sid(sid)

                if found_player:
                    return found_player, game, room_id

            if found_player: break

        else:
            return None, None, None


    def remove_room(self, room_id):
        for game_type_rooms in self.rooms.values():
            if room_id in game_type_rooms:
                del game_type_rooms[room_id]


    def start_model_service(self, label_pair, room_id):
        model_service_base = Settings.get("MODEL_SERVICE_BASE", "http://localhost:5002")
        requests.post(
            f"{model_service_base}/start_model",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"label_pair": label_pair}),
        )


    def stop_model_service(self, label_pair):
        model_service_base = Settings.get("MODEL_SERVICE_BASE", "http://localhost:5002")
        requests.post(
            f"{model_service_base}/stop_model",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"label_pair": label_pair}),
        )
