from typing import Union

from competitive_drawing.web_app.game import GameType

from ..game import Game, Player


class FreePlayGame(Game):
    def __init__(self, *args, **kwargs):
        self.game_type = GameType.FREE_PLAY
        raise NotImplementedError("Free play is not implemented")


    @property
    def can_start_game(self):
        return len(self.players) >= 1


    def add_player(self, sid: Union[str, None]) -> Player:
        return super().add_player(sid)
