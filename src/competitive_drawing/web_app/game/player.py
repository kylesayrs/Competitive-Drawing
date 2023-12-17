from dataclasses import dataclass


@dataclass
class Player():
    """
    Player model for storing player identifiers and attributes

    :param id: hex string which uniquely identifies the player within the room
    :param sid: session id of the room to which the player belongs
    :param target: name of the the player's target drawing class
    :param target_index: index of classifier output which corresponds to target
    """
    id: str
    sid: str
    target: str
    target_index: int
