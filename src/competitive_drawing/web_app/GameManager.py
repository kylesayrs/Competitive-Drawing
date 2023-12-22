from typing import List, Tuple, Union, Optional

from PIL import Image
import json
import random
import requests
from collections import defaultdict

from competitive_drawing import Settings
from .game import GameType, Game, Player, create_game
from .sockets import emit_end_game
from .model_service import server_infer, server_update


class GameManager:
    """
    Handles the creation and deletion of games as well as the assignment
    of players to games
    """
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
            if not game.started
        ]

        # if there are rooms needing players, add to room
        if len(available_game_room_ids) > 0:
            random.shuffle(available_game_room_ids)
            return available_game_room_ids[0]

        # otherwise create a new game and room
        new_game = self._new_game(game_type, *game_args, **game_kwargs)
        return new_game.room_id

    
    def join_game(self, room_id: str, player_sid: str, player_id_cache: Union[str, None]):
        """
        Called when a player joins a game. Assigns player to game and starts game
        if applicable

        :param room_id: room id being joined
        :param player_sid: socket session id
        :param player_id_cache: player id stored in client storage
        """
        game = self.game_by_room_id[room_id]

        if game is None:
            print(f"WARNING: Could not find game associated with room id {room_id}")
            return
        
        if not game.started:
            # add player
            new_player = game.add_player(player_sid)
            self.player_game_by_sid[player_sid] = (new_player, game)

            # start game
            if game.can_start_game:
                game.start_game()

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


    def end_turn(
        self,
        room_id: str,
        player_id_cache: Union[str, None],
        canvas_data_url: Union[str, None],
        canvas_preview_data_url: Union[str, None]
    ):
        game = self.game_by_room_id[room_id]

        if game is None:
            raise ValueError(
                f"WARNING: Could not find game associated with room id {room_id}"
            )
        
        if player_id_cache is None:
            raise ValueError(
                f"WARNING: Unknown player with id {player_id_cache} tried to end "
                "turn"
            )
        
        if canvas_data_url is None:
            raise ValueError(
                f"WARNING: Player with id {player_id_cache} tried to end turn "
                "without an image"
            )
        
        if canvas_preview_data_url is None:
            raise ValueError(
                f"WARNING: Player with id {player_id_cache} tried to end turn "
                "without a preview image"
            )
                
        if player_id_cache != game.turn.id:
            raise ValueError(
                "WARNING: Received wrong player id for end turn "
                f"({player_id_cache} != {game.turn.id})"
            )
        
        # save image
        # TODO: image for future data mining
        game.next_turn(canvas_data_url, canvas_preview_data_url)

        if game.can_end_game:
            self.end_game(game, canvas_preview_data_url)


    def end_game(
        self,
        game: Game,
        canvas_preview_data_url: Optional[str] = None,
        force_loser: Optional[Player] = None
    ):
        game.end_game(canvas_preview_data_url, force_loser)
        self._del_game(game)


    def _new_game(self, game_type: GameType, *game_args, **game_kwargs) -> Game:
        """
        Create a new game

        :param game_type: type of game being created
        :param game_args: arguments used to initialize new game
        :param game_kwargs: keyword arguments used to initialize new game
        :return: instance of new game
        """
        # initialize new game
        new_game = create_game(game_type, *game_args, **game_kwargs)

        # update indexes
        self.games.append(new_game)
        self.game_by_room_id[new_game.room_id] = new_game
        self.games_by_gametype[new_game.game_type].append(new_game)
        self.games_by_label_pair_str[new_game.label_pair_str].append(new_game)

        # communicate games information to model service
        self._update_model_service()

        return new_game
    

    def _del_game(self, game: Game):
        """
        Delete a game instance after the game is finished

        :param game: game to be deleted
        """
        # update indexes
        self.games.remove(game)
        del self.game_by_room_id[game.room_id]
        self.games_by_gametype[game.game_type].remove(game)
        self.games_by_label_pair_str[game.label_pair_str].remove(game)

        for player in game.players:
            if player.sid is not None:
                del self.player_game_by_sid[player.sid]

        # communicate games information to model service
        self._update_model_service()

        del game  # redundancy
        

    def _update_model_service(self):
        """
        Communicate information about the number of active games and label pairs
        so model service can make scaling decisions
        """
        num_games_by_label_pair_str = {
            label_pair_str: len(self.games_by_label_pair_str[label_pair_str])
            for label_pair_str in self.games_by_label_pair_str
        }

        server_update(num_games_by_label_pair_str)
