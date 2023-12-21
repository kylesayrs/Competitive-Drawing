from ..game import GameType, FreePlayGame, LocalGame, OnlineGame, SinglePlayerGame


def create_game(game_type: GameType, *game_args, **game_kwargs):
    match game_type:
        case GameType.FREE_PLAY:
            return FreePlayGame(*game_args, **game_kwargs)
        
        case GameType.LOCAL:
            return LocalGame(*game_args, **game_kwargs)

        case GameType.ONLINE:
            return OnlineGame(*game_args, **game_kwargs)

        case GameType.SINGLE_PLAYER:
            return SinglePlayerGame(*game_args, **game_kwargs)

        case _:
            raise ValueError(f"Unimplemented game type {game_type.value}")
