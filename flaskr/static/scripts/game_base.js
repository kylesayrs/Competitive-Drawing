// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/drawing_board.js";

// global state
const allLabels = ["sheep", "dragon", "mona_lisa", "guitar", "pig",
             "tree", "clock", "squirrel", "duck", "jail"]
var targetLabels = ["pig", "duck"];
var inferenceMutex = false;

const confidenceChart = new ConfidenceChart(targetLabels)
const distanceIndicator = new DistanceIndicator(150, 0)
const drawingBoard = new DrawingBoard(distanceIndicator)

// initialize model inference session
// TODO: wrap this in a promise and have game wait until model is initialized
const inferenceSessionPromise = ort.InferenceSession.create(
    "/static/models/model.onnx"
);
inferenceSessionPromise.then(() => {
    console.log("Loaded ort")
})

function softmax(arr, factor=1) {
    const exponents = arr.map((value) => Math.exp(value * factor))
    const total = exponents.reduce((a, b) => a + b, 0);
    return exponents.map((exp) => exp / total);
}

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

async function clientInferImage(callbackFn=null) {
    drawingBoard.updatePreview(async (previewImageData) => {
        // grab data from preview canvas
        const imageDataBuffer = imageData2BWData(previewImageData)

        // create input
        const model_input = new ort.Tensor(
            imageDataBuffer,
            [1, 1, 28, 28]
        );

        // perform inference
        // TODO: remove promise when game loads after inference session is loaded
        const inferenceSession = await inferenceSessionPromise
        const model_outputs = await inferenceSession.run({ "input": model_input })

        // normalize scores
        const model_outputs_normalized = normalize(model_outputs.output.data, 0, 1)

        // filter to target outputs
        var filteredLabels = []
        var filteredOutputs = []
        for (let i = 0; i < allLabels.length; i++) {
            if (targetLabels.includes(allLabels[i])) {
                filteredLabels.push(allLabels[i])
                filteredOutputs.push(model_outputs_normalized[i])
            }
        }

        // apply softmax
        const model_confidences = softmax(filteredOutputs, 7)

        // update chart
        var chartData = []
        for (let i = 0; i < filteredLabels.length; i++) {
            chartData.push({"label": filteredLabels[i], "value": model_confidences[i]})
        }
        confidenceChart.updateData(chartData)

        if (callbackFn) {
            callbackFn()
        }
    })
}

async function serverInferImage(callbackFn=null) {
    return; // TODO: implement

    response = await fetch(
        inferenceUrl,
        {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": JSON.stringify({"imageData": canvasImageData})
        }
    )
    if (callbackFn) {
        callbackFn()
    }
}

drawingBoard.afterMouseEnd = async () => {
    // TODO: await on mutex
    if (!inferenceMutex) {
        inferenceMutex = true;
        clientInferImage(() => {
            inferenceMutex = false;
        })
    }

    serverInferImage()
}

drawingBoard.afterMouseMove = async () => {
    if (!inferenceMutex) {
        inferenceMutex = true;
        await clientInferImage(() => {
            inferenceMutex = false;
        })
    }
}

//TODO: window.onresize = () =>
