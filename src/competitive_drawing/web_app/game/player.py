from typing import Union

from dataclasses import dataclass


@dataclass
class Player():
    """
    Player model for storing player identifiers and attributes

    :param id: hex string which uniquely identifies the player within the room
    :param sid: player's socket session id used to track connects and disconnects
    :param target: name of the the player's target drawing class
    :param target_index: index of classifier output which corresponds to target
    """
    id: str
    sid: Union[str, None]
    target: str
    target_index: int
