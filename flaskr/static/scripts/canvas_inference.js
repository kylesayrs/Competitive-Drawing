function normalize(arr, minNorm=0, maxNorm=1) {
    const minimumValue = Math.min.apply(Math, arr)
    arr = arr.map((value) => value - minimumValue, minNorm)
    const maxValue = Math.max.apply(Math, arr)
    const ratio = maxValue * maxNorm

    for (let i = 0; i < arr.length; i++ ) {
        arr[i] /= ratio;
    }
    return arr
}

function imageData2BWData(imageData) {
    // need to get alpha channel because MarvinJ's getColorComponent is broken
    var alphaChannelBuffer = []
    for (let i = 0; i < imageData.data.length; i += 4) {
        alphaChannelBuffer.push(imageData.data[i + 3]);//resizedImageData[i + 2])
    }

    return new Float32Array(alphaChannelBuffer)
}

function softmax(arr, factor=1) {
    const exponents = arr.map((value) => Math.exp(value * factor))
    const total = exponents.reduce((a, b) => a + b, 0);
    return exponents.map((exp) => exp / total);
}

export class Inferencer {
    constructor(allLabels, targetLabels) {
        this.allLabels = allLabels
        this.targetLabels = targetLabels

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
        const model_outputs = await (await this.inferenceSession).run({ "input": model_input })

        // normalize scores
        const model_outputs_normalized = normalize(model_outputs.output.data, 0, 1)

        // filter to target outputs
        var filteredOutputs = []
        for (let i = 0; i < this.allLabels.length; i++) {
            if (this.targetLabels.includes(this.allLabels[i])) {
                filteredOutputs.push(model_outputs_normalized[i])
            }
        }

        // apply softmax
        const model_confidences = softmax(filteredOutputs, 7)

        if (callbackFn) {
            callbackFn(model_confidences)
        } else {
            return model_confidences
        }
    }
}
