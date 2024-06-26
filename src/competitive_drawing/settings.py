from threading import Lock
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global settings for running the web app and model service
    """

    # web app
    web_app_host: str = Field(default="localhost")
    web_app_port: int = Field(default=5001)
    web_app_secret_key: str = Field(default="somesecrets")
    ms_base: str = Field(default="http://localhost:5002")

    # model service
    ms_host: str = Field(default="localhost")
    ms_port: int = Field(default=5002)
    ms_secret_key: str = Field(default="somesecrets")
    ws_base: str = Field(default="http://localhost:5001")
    device: str = Field(default="cpu")

    # game settings
    softmax_factor: float = Field(default=2.0)
    distance_per_turn: int = Field(
        default=40,
        description="distance pen can travel per turn. Measured in canvas-pixels"
    )
    total_num_turns: int = Field(
        default=10,
        description=(
            "total number of turns shared between players. Should be an even "
            "number for fairness"
        )
    )

    # visualization settings
    canvas_size: int = Field(default=100)
    image_size: int = Field(default=50)
    image_padding: int = Field(default=0)
    canvas_line_width: float = Field(
        default=1.5,
        description="width of stroke on canvas in pixels"
    )
    static_crop: bool = Field(
        default=True,
        description=(
            "False if image should be cropped to strokes on canvas, True otherwise"
        )
    )

    # aws settings
    s3_models_bucket: str = Field(default="competitive-drawing-models-prod")
    s3_models_root_folder: str = Field(default="static_crop_50x50")
    s3_model_duration: int = Field(default=108000, description="lifetime of s3 model urls in seconds")  # 30 minutes

    # websocket settings
    client_disconnect_grace_period: float = Field(
        default=2.0,
        description=(
            "number of seconds after a client disconnects before they leave "
            "the room"
        )
    )

    # make model immutable
    class Config:
        frozen = True


SETTINGS = Settings()
