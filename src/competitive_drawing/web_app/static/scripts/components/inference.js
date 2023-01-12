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


    async serverInferImage(imageDataUrl, targetIndex) {
        const response = await fetch(
            "/infer",
            {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": JSON.stringify({
                    "gameConfig": this.gameConfig,
                    "label_pair": this.label_pair,
                    "imageDataUrl": imageDataUrl,
                })
            }
        )
        if (!response.ok) {
            console.log("Invalid server inference response")
        }

        const responseJson = await response.json()
        return responseJson["modelOutputs"]
    }


    async serverInferImageWithGrad(imageDataUrl, targetIndex) {
        const response = await fetch(
            "/infer",
            {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": JSON.stringify({
                    "imageDataUrl": imageDataUrl,
                    "targetIndex": targetIndex,
                })
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


    async serverInferStroke(imageDataUrl, targetIndex) {
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
                })
            }
        )
        if (!response.ok) {
            console.log("Invalid server inference response")
        }

        const responseJson = await response.json()
        return responseJson["strokeSamples"]
    }
}
