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
    constructor(gameConfig, targets) {
        this.gameConfig = gameConfig
        this.label_pair = Object.values(targets).sort()
    }


    async loadModel(modelUrl, imageSize=50) {
        this.imageSize = imageSize

        return new Promise((resolve) => {
            this.inferenceSession = ort.InferenceSession.create(modelUrl);
            this.inferenceSession.then(() => {
                console.log("Loaded ort")
                resolve()
            })
        })
    }


    async clientInferImage(previewImageData) {
        // get from preview image data
        const modelInputData = imageDataToModelInputData(previewImageData)

        // create input
        const modelInput = new ort.Tensor(
            modelInputData,
            [1, 1, this.imageSize, this.imageSize]
        );

        // perform inference
        const modelOutputsRaw = await (await this.inferenceSession).run({ "input": modelInput })
        const modelOutputs = modelOutputsRaw.logits.data

        return modelOutputs
    }


    async serverInferStroke(imageDataUrl, targetIndex, roomId) {
        const response = await fetch(
            "/infer_stroke",
            {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": JSON.stringify({
                    "gameConfig": this.gameConfig,
                    "label_pair": this.label_pair,
                    "targetIndex": targetIndex,
                    "imageDataUrl": imageDataUrl,
                    "roomId": roomId,
                })
            }
        )
        if (!response.ok) {
            console.log("Invalid server stroke inference response")
        }
    }
}
