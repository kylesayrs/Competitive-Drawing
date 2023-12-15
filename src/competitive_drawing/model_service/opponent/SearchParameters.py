from pydantic import BaseModel, Field


class SearchParameters(BaseModel):
    """
    Data class which specifies parameters for searching strokes 

    :param BaseModel: _description_
    """
    num_keypoints: int = Field(default=4)
    max_width: float = Field(default=10.0)
    min_width: float = Field(default=3.5)
    max_aa: float = Field(default=0.35)
    min_aa: float = Field(default=0.9)
    max_steps: int = Field(default=125)
    return_best: bool = Field(default=False)
    draw_output: bool = Field(default=False)
