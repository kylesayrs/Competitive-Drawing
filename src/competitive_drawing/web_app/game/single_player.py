from typing import Union

from competitive_drawing.web_app.game import GameType

from ..game import Game, Player
from ..sockets import emit_assign_player
from ..model_service import server_infer_ai


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
    

    def next_turn(self, canvas_data_url: str, preview_data_url: str):
        super().next_turn(canvas_data_url, preview_data_url)

        if self._player_turn_index == 1: # now AI player's turn
            server_infer_ai(
                self.label_pair,
                preview_data_url,
                1,
                self.room_id
            )
