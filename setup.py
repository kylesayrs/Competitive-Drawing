from setuptools import setup, find_packages

_deps = [
    "Flask",
    "Flask-SocketIO",
    "Flask-Cors",
    "grad-cam",
    "numpy",
    "onnx",
    "Pillow",
    "torch",
    "tqdm",
    "wandb",
    "boto3",
    "simple-websocket"
]

setup(
    name="competitive_drawing",
    version="1.0",
    author="Kyle Sayers",
    description="",
    install_requires=_deps,
    package_dir={"": "src"},
    packages=find_packages("src", include=["competitive_drawing"], exclude=["*.__pycache__.*"]),
    entry_points={
        "console_scripts": [
            "competitive_drawing.launch_model_service = competitive_drawing.model_service.app:start_app",
            "competitive_drawing.launch_web_app = competitive_drawing.web_app.app:start_app",
            "competitive_drawing.train_all_models = competitive_drawing.train.train_all_models:main",
        ],
    },
)
