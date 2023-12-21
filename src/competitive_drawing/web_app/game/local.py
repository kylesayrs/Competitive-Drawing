from typing import Union

from competitive_drawing.web_app.game import GameType

from ..game import Game, Player
from ..sockets import emit_assign_player


class LocalGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_type = GameType.LOCAL


    @property
    def can_start_game(self):
        return len(self.players) >= 2


    def add_player(self, sid: Union[str, None]) -> Player:
        player_one = super().add_player(sid)
        player_two = super().add_player(None)  # virtual player 2
        emit_assign_player(player_one.id, sid)
        emit_assign_player(player_two.id, sid)

        return player_one
