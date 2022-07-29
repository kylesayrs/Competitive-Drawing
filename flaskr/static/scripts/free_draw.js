// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/drawing_board.js";
import { Inferencer } from "/static/scripts/canvas_inference.js";

// global state
const allLabels = ["sheep", "dragon", "mona_lisa", "guitar", "pig",
             "tree", "clock", "squirrel", "duck", "jail"]

const confidenceChart = new ConfidenceChart(allLabels)
const drawingBoard = new DrawingBoard()
const inferencer = new Inferencer(allLabels)

var inferenceMutex = false // true for locked, false for unlocked

async function clientInferImage() {
    if (inferenceMutex) { return }
    inferenceMutex = true

    const previewImageData = await drawingBoard.updatePreview()
    const model_confidences = await inferencer.inferPreviewImageData(previewImageData)

    // update chart
    var chartData = []
    for (let i = 0; i < allLabels.length; i++) {
        chartData.push({"label": allLabels[i], "value": model_confidences[i]})
    }
    confidenceChart.updateData(chartData)
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
