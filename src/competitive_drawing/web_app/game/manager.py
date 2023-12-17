from typing import List, Dict, Tuple, Union

import uuid
import json
import random
import requests

from competitive_drawing import Settings
from ..game import GameType, Game, Player


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
            if not game.started and game.can_add_player
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
            self.start_model_service(new_game.label_pair)

        if new_game.label_pair not in self.label_pair_rooms:
            self.label_pair_rooms[new_game.label_pair] = []

        self.label_pair_rooms[new_game.label_pair].append(new_room_id)

        return new_room_id


    def get_game(self, game_type: GameType, room_id: str) -> Game:
        return self.rooms[game_type][room_id]


    def has_label_pair(self, label_pair: Tuple[str, str]) -> bool:
        for game_type_rooms in self.rooms.values():
            for room_id, game in game_type_rooms.items():
                if game.label_pair == label_pair:
                    return True

        return False


    def get_player_location_by_sid(
        self,
        sid: str
    ) -> Union[Tuple[Player, Game, str], Tuple[None, None, None]]:
        found_player = None
        for game_type_rooms in self.rooms.values():
            for room_id, game in game_type_rooms.items():
                found_player = game.get_player_by_sid(sid)

                if found_player:
                    return found_player, game, room_id

            if found_player: break

        else:
            return None, None, None


    def remove_room(self, room_id: str):
        for game_type_rooms in self.rooms.values():
            if room_id in game_type_rooms:
                del game_type_rooms[room_id]


    def start_model_service(self, label_pair: Tuple[str, str]):
        requests.post(
            f"{Settings().model_service_base}/start_model",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"label_pair": label_pair}),
        )


    def stop_model_service(self, label_pair):
        requests.post(
            f"{Settings().model_service_base}/stop_model",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"label_pair": label_pair}),
        )
