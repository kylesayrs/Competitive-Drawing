/*
filename: canvas_inference.js
author: Kyle Sayers
details: The Inferencer is responsible for running the model from canvas data
*/

function imageDataToModelInputData(imageData) {
    // TODO use marvinj
    var redChannelBuffer = []
    for (let i = 0; i < imageData.data.length; i += 4) {
        redChannelBuffer.push(1 - (imageData.data[i + 0] / 255))
    }

    return new Float32Array(redChannelBuffer);
}

export class Inferencer {
    constructor(gameConfig, targets, onnxUrl) {
        this.gameConfig = gameConfig
        this.label_pair = Object.values(targets).sort()
        this.onnxUrl = onnxUrl

        this.inferenceSession = null
    }

    
    async initialize() {
        if (!this.inferenceSession) {
            this.inferenceSession = ort.InferenceSession.create(this.onnxUrl);
        }

        return this.inferenceSession
    }


    async clientInferImage(previewImageData) {
        // get from preview image data
        const modelInputData = imageDataToModelInputData(previewImageData)

        // create input
        const modelInput = new ort.Tensor(
            modelInputData,
            [1, 1, this.gameConfig.imageSize, this.gameConfig.imageSize]
        );

        // perform inference
        const modelOutputsRaw = await (await this.inferenceSession).run({ "input": modelInput })
        const modelOutputs = modelOutputsRaw.logits.data

        return modelOutputs
    }
}
