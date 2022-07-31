/*
filename: canvas_inference.js
author: Kyle Sayers
details: The Inferencer is responsible for running the model from canvas data
*/

function imageData2BWData(imageData) {
    // need to get alpha channel because MarvinJ's getColorComponent is broken
    var alphaChannelBuffer = []
    for (let i = 0; i < imageData.data.length; i += 4) {
        alphaChannelBuffer.push(imageData.data[i + 3]);//resizedImageData[i + 2])
    }

    return new Float32Array(alphaChannelBuffer)
}

export class Inferencer {
    constructor() {
        this.inferenceSession = ort.InferenceSession.create(
            "/static/models/model.onnx"
        );
        this.inferenceSession.then(() => console.log("Loaded ort"))
    }

    async inferPreviewImageData(previewImageData, callbackFn=null) {
        // grab data from preview canvas
        const imageDataBuffer = imageData2BWData(previewImageData)

        // create input
        const model_input = new ort.Tensor(
            imageDataBuffer,
            [1, 1, 28, 28]
        );

        // perform inference
        // TODO: remove promise when game loads after inference session is loaded
        const modelOutputs = await (await this.inferenceSession).run({ "input": model_input })

        if (callbackFn) {
            callbackFn(modelOutputs)
        } else {
            return modelOutputs
        }
    }
}
