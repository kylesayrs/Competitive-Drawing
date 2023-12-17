from typing import List, Tuple, Union, Optional

from PIL import Image
import json
import numpy
import random
import requests
from collections import defaultdict

from competitive_drawing import Settings
from ..game import GameType, Game, Player
from ..sockets import emit_assign_player, emit_start_game, emit_start_turn, emit_end_game
from ..utils import data_url_to_image, get_game_config


class GameManager:
    games: List[Game]

    # game indices
    game_by_room_id: defaultdict[str, Union[Game, None]]
    games_by_gametype: defaultdict[GameType, List[Game]]
    games_by_label_pair_str: defaultdict[str, List[Game]]

    # player indices
    player_game_by_sid: defaultdict[str, Union[Tuple[Player, Game], Tuple[None, None]]]


    def __init__(self):
        self.games = []

        self.game_by_room_id = defaultdict(lambda: None)
        self.games_by_gametype = defaultdict(lambda: [])
        self.games_by_label_pair_str = defaultdict(lambda: [])

        self.player_game_by_sid = defaultdict(lambda: (None, None))


    def assign_game_room(self, game_type: GameType, *game_args, **game_kwargs) -> str:
        """
        Assigns a new player a game room to join based on their desired game type

        :param game_type: desired game type of room
        :param game_args: arguments used to initialize new game if necessary
        :param game_kwargs: keyword arguments used to initialize new game if necessary
        :return: assigned room id
        """
        available_game_room_ids = [
            game.room_id
            for game in self.games_by_gametype[game_type]
            if not game.started and game.can_add_player
        ]

        # if there are rooms needing players, add to room
        if len(available_game_room_ids) > 0:
            random.shuffle(available_game_room_ids)
            return available_game_room_ids[0]

        # otherwise create a new game and room
        new_game = self.new_game(game_type, *game_args, **game_kwargs)
        return new_game.room_id


    def new_game(self, game_type: GameType, *game_args, **game_kwargs) -> Game:
        # initialize new game
        new_game = Game(game_type, *game_args, **game_kwargs)

        # update indexes
        self.games.append(new_game)
        self.game_by_room_id[new_game.room_id] = new_game
        self.games_by_gametype[new_game.game_type] += [new_game]
        self.games_by_label_pair_str[new_game.label_pair_str] += [new_game]

        # communicate games information to model service
        #TODO: replace with self.update_model_service()
        self.start_model_service(new_game.label_pair)

        return new_game
    

    def del_game(self, game: Game):
        # update indexes
        self.games.remove(game)
        del self.game_by_room_id[game.room_id]
        self.games_by_gametype[game.game_type].remove(game)
        self.games_by_label_pair_str[game.label_pair_str].remove(game)

        for player in game.players:
            if player.sid is not None:
                del self.player_game_by_sid[player.sid]

        del game  # redundancy

    

    def join_game(self, room_id: str, player_sid: str, player_id_cache: Union[str, None]):
        game = self.game_by_room_id[room_id]

        if game is None:
            print(f"WARNING: Could not find game associated with room id {room_id}")
            return
        
        if not game.started:
            # add player(s) to game
            if game.game_type == GameType.FREE_PLAY:
                print("WARNING: free play is not implemented yet")
                return

            elif game.game_type == GameType.LOCAL:
                player_one = game.add_player(player_sid)
                player_two = game.add_player(None)
                emit_assign_player(player_one.id)
                emit_assign_player(player_two.id)

                self.player_game_by_sid[player_sid] = (player_one, game)

            elif game.game_type == GameType.ONLINE:
                if game.can_add_player():
                    new_player = game.add_player(player_sid)
                    emit_assign_player(new_player.id)

                    self.player_game_by_sid[player_sid] = (new_player, game)

            elif game.game_type == GameType.SINGLE_PLAYER:
                player_one = game.add_player(player_sid)
                player_two = game.add_player(None)  # AI opponent

                self.player_game_by_sid[player_sid] = (player_one, game)

            else:
                print(f"WARNING: Unknown game type {game.game_type}")
                return

            # start game
            if game.can_start_game:
                game.start_game()
                emit_start_game(game, room_id)
                emit_start_turn(game, room_id)

        else:
            if player_id_cache is None:
                print(
                    f"WARNING: tried to join started room {room_id} but no cached "
                    "player id was found"
                )
                return

            if not game.has_player(player_id_cache):
                print(
                    f"WARNING: Player with id {player_id_cache} tried to join game "
                    f"with players {game.players}"
                )
                return

            # assume a disconnect: game is resumed
            game.reassign_player_sid(player_id_cache, player_sid)
            emit_start_game(game, room_id)
            emit_assign_player(player_id_cache)
            emit_start_turn(game, room_id)


    def end_turn(
        self,
        room_id: str,
        player_id_cache: Union[str, None],
        canvas_data_url: Union[str, None],
        canvas_preview_data_url: Union[str, None]
    ):
        game = self.game_by_room_id[room_id]

        if game is None:
            print(f"WARNING: Could not find game associated with room id {room_id}")
            return
        
        if player_id_cache is None:
            print(f"WARNING: Unknown player with id {player_id_cache} tried to end turn")
            return
        
        if canvas_data_url is None:
            print(f"WARNING: Player with id {player_id_cache} tried to end turn without an image")
            return
        
        if canvas_preview_data_url is None:
            print(f"WARNING: Player with id {player_id_cache} tried to end turn without a preview image")
            return
                
        if player_id_cache != game.turn.id:
            print(f"WARNING: Received wrong player id for end turn ({player_id_cache} != {game.turn.id})")
            return

        # save image
        # TODO: image for future data mining
        canvas_image = data_url_to_image(canvas_data_url)
        game.next_turn(canvas_image)

        if game.can_end_game():
            self.end_game(game, canvas_preview_data_url)
        else:
            emit_start_turn(game, room_id)


    def end_game(
        self,
        game: Game,
        canvas_preview_data_url: Optional[str] = None,
        force_loser: Optional[Player] = None
    ):
        assert canvas_preview_data_url is not None or force_loser is not None
        if canvas_preview_data_url:
            # do another inference for redundancy
            response = requests.post(
                f"{Settings().model_service_base}/infer",
                headers={ "Content-Type": "application/json" },
                data=json.dumps({
                    "gameConfig": get_game_config(),
                    "label_pair": game.label_pair,
                    "imageDataUrl": canvas_preview_data_url
                })
            )

            # emit winner
            winner_index = int(response.json()["modelOutputs"][1] > response.json()["modelOutputs"][0])
            winner_target = game.label_pair[winner_index]
            emit_end_game(winner_target, game.room_id)

        else:
            winner = game.get_other_player(force_loser)
            emit_end_game(winner.target, game.room_id)

        # delete game
        self.del_game(game)
    

    def set_canvas_image(self, room_id: str, canvas_image: Image):
        game = self.game_by_room_id[room_id]
        game.canvasImage = canvas_image
        if game is None:
            print(f"WARNING: Could not find game associated with room id {room_id}")
            return


    def start_model_service(self, label_pair: Tuple[str, str]):
        requests.post(
            f"{Settings().model_service_base}/start_model",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"label_pair": label_pair}),
        )


    def stop_model_service(self, label_pair: Tuple[str, str]):
        requests.post(
            f"{Settings().model_service_base}/stop_model",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"label_pair": label_pair}),
        )
