from typing import Dict, List

from setuptools import setup, find_packages

# TODO: loosen these versions
_deps = [
    "Flask==3.0.3",
    "Flask-SocketIO==5.3.2",
    "Flask-Cors==3.0.10",
    "grad-cam==1.4.6",
    "numpy",
    "onnx",
    "Pillow",
    "torch",
    "tqdm",
    "boto3==1.26.43",
    "simple-websocket==0.9.0",
    "pydantic==2.7.4",
    "pydantic_settings==2.3.3",
]

_dev_deps = [
    "wandb",
    "sparseml",
    "cairocffi",
    "pytest"
]

def _setup_extras() -> Dict[str, List[str]]:
    return {
        "dev": _dev_deps
    }


setup(
    name="competitive_drawing",
    version="1.0",
    author="Kyle Sayers",
    description="",
    install_requires=_deps,
    package_dir={"": "src"},
    packages=find_packages("src", include=["competitive_drawing"], exclude=["*.__pycache__.*"]),
    extras_require=_setup_extras(),
    entry_points={
        "console_scripts": [
            "competitive_drawing.launch_model_service = competitive_drawing.model_service.app:start_app",
            "competitive_drawing.launch_web_app = competitive_drawing.web_app.app:start_app",
            "competitive_drawing.train_all_models = competitive_drawing.train.train_all_models:main",
        ],
    },
)
