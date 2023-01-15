from typing import Optional, Any

class Settings:
    WEB_APP_HOST="localhost"
    WEB_APP_PORT=5001
    WEB_APP_SECRET_KEY="somesecrets"
    MODEL_SERVICE_BASE="http://localhost:5002"

    MODEL_SERVICE_HOST="localhost"
    MODEL_SERVICE_PORT=5002
    MODEL_SERVICE_SECRET_KEY="somesecrets"
    ALLOWED_ORIGIN="localhost:5001"

    SOFTMAX_FACTOR=2.0
    DISTANCE_PER_TURN=30
    TOTAL_NUM_TURNS=10  # should be an even number for fairness

    CANVAS_SIZE=100
    IMAGE_SIZE=50
    IMAGE_PADDING=0
    CANVAS_LINE_WIDTH=1.5
    STATIC_CROP=1

    S3_MODELS_BUCKET="competitive-drawing-models-prod"
    S3_MODELS_ROOT_FOLDER="static_crop_50x50"
    S3_MODEL_URL_DURATION=108000  # 30 minutes in seconds
    S3_MODEL_DURATION=108000  # 30 minutes in seconds

    PAGE_REFRESH_BUFFER_TIME=2.0  # a client has 2 seconds to reconnect after a disconnect


    @classmethod
    def get(cls, key: str, default_value: Optional[Any] = None):
        if hasattr(cls, key):
            return getattr(cls, key)

        elif default_value is not None:
            print(f"Warning: falling back on default value for key {key}")
            return default_value

        else:
            raise ValueError(f"Could not find setting for {key}")
