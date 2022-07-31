// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/canvas_inference.js";

// global state
const allLabels = ["sheep", "dragon", "mona_lisa", "guitar", "pig",
             "tree", "clock"]
const targetLabels = ["clock", "guitar"];

const confidenceChart = new ConfidenceChart(allLabels, targetLabels)
const distanceIndicator = new DistanceIndicator(150, 0)
const drawingBoard = new DrawingBoard(distanceIndicator)
const inferencer = new Inferencer()

var inferenceMutex = false // true for locked, false for unlocked

async function clientInferImage() {
    if (inferenceMutex) { return }
    inferenceMutex = true

    const previewImageData = await drawingBoard.updatePreview()
    const modelOutputs = await inferencer.inferPreviewImageData(previewImageData)

    confidenceChart.update(modelOutputs)
    inferenceMutex = false
}

async function serverInferImage() {
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
}

drawingBoard.afterMouseEnd = async () => {
    clientInferImage()
    serverInferImage()
}

drawingBoard.afterMouseMove = async () => {
    clientInferImage()
}

//TODO: window.onresize = () =>
