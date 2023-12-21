from typing import Union

from competitive_drawing.web_app.game import GameType

from ..game import Game, Player
from ..sockets import emit_assign_player


class SinglePlayerGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_type = GameType.SINGLE_PLAYER


    @property
    def can_start_game(self):
        return len(self.players) >= 1


    def add_player(self, sid: Union[str, None]) -> Player:
        player_one = super().add_player(sid)
        player_two = super().add_player(None)  # AI opponent

        emit_assign_player(player_one.id, sid)

        return player_one
