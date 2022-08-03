/*
filename: canvas_inference.js
author: Kyle Sayers
details: The Inferencer is responsible for running the model from canvas data
*/

function imageDataToModelInputData(imageData) {
    // need to get alpha channel because MarvinJ's getColorComponent is broken
    var alphaChannelBuffer = []
    for (let i = 0; i < imageData.data.length; i += 4) {
        alphaChannelBuffer.push(imageData.data[i + 3] / 255);
    }

    return new Float32Array(alphaChannelBuffer);
}

export class Inferencer {
    constructor() {
        this.inferenceSession = ort.InferenceSession.create(
            "/static/models/model.onnx"
        );
        this.inferenceSession.then(() => console.log("Loaded ort"))
    }

    async serverInferImage(imageDataUrl) {
        const response = await fetch(
            "/infer",
            {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": JSON.stringify({"imageDataUrl": imageDataUrl})
            }
        )
        if (!response.ok) {
            console.log("Invalid server inference response")
        }
        const responseJson = await response.json()
        const modelOutputs = responseJson["modelOutputs"]
        const gradCamImage = responseJson["gradCamImage"]
        return { modelOutputs, gradCamImage }
    }

    async clientInferImage(previewImageData) {
        // get from preview image data
        const modelInputData = imageDataToModelInputData(previewImageData)

        // create input
        const modelInput = new ort.Tensor(
            modelInputData,
            [1, 1, 28, 28]
        );

        // perform inference
        const modelOutputsRaw = await (await this.inferenceSession).run({ "input": modelInput })
        const modelOutputs = modelOutputsRaw.logits.data

        return modelOutputs
    }
}
