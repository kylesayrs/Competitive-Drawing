from typing import Union

from competitive_drawing.web_app.game import GameType

from ..game import Game, Player
from ..sockets import emit_assign_player


class OnlineGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_type = GameType.ONLINE


    @property
    def can_start_game(self):
        return len(self.players) >= 2


    def add_player(self, sid: Union[str, None]) -> Player:
        new_player = super().add_player(sid)
        emit_assign_player(new_player.id, sid)

        return new_player
