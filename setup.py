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
]

setup(
    name="drawnt",
    version="1.0",
    author="Kyle Sayers",
    description="",
    install_requires=_deps,
    package_dir={"": "src"},
    packages=find_packages("src", include=["drawnt"], exclude=["*.__pycache__.*"]),
    entry_points={
        "console_scripts": [
            "drawnt.launch_model_service = drawnt.model_service.app:start_app",
            "drawnt.launch_web_app = drawnt.web_app.app:start_app",
        ],
    },
)
