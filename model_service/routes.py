# canvas to numpy (to be moved)
from io import BytesIO, StringIO
from PIL import Image
import re


@route.route("/infer", methods=["POST"])
def infer():
    # TODO: move this to utils file
    image_data_url = request.json["imageDataUrl"]
    image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
    image_data = base64.b64decode(image_data_str)
    image_data_io = BytesIO(image_data)
    image = Image.open(image_data_io)

    target_index = request.json["targetIndex"]

    #model_outputs = inferencer.infer_image(image)
    model_outputs, grad_cam_image = inferencer.infer_image_with_cam(image, target_index)

    # TODO cheat detection

    return route.response_class(
        response=json.dumps({
            "modelOutputs": model_outputs,
            "gradCamImage": grad_cam_image,
            "isCheater": False,
        }),
        status=200,
        mimetype="routelication/json"
    )