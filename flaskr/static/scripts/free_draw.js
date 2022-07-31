// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/canvas_inference.js";

// global state
const allLabels = ["sheep", "dragon", "mona_lisa", "guitar", "pig",
             "tree", "clock", "squirrel", "duck", "jail"]

const confidenceChart = new ConfidenceChart(allLabels)
const drawingBoard = new DrawingBoard()
const inferencer = new Inferencer()

var inferenceMutex = false // true for locked, false for unlocked

async function clientInferImage() {
    if (inferenceMutex) { return }
    inferenceMutex = true

    const previewImageData = await drawingBoard.updatePreview()
    const modelOutputs = await inferencer.inferPreviewImageData(previewImageData)

    // update chart
    confidenceChart.update(modelOutputs)
    inferenceMutex = false
}

async function serverInferImage() {
    return
    const canvasBlobUrl = drawingBoard.canvas.toDataURL();
    const response = await fetch(
        "/infer",
        {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": JSON.stringify({"canvasBlobUrl": canvasBlobUrl})
        }
    )
    if (!response.ok) {
        console.log("Invalid server inference response")
    }
    const responseJson = await response.json()
    console.log(responseJson)
}

drawingBoard.afterMouseEnd = async () => {
    clientInferImage()
    serverInferImage()
}

drawingBoard.afterMouseMove = async () => {
    clientInferImage()
}

//TODO: window.onresize = () =>
