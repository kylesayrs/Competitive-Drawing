// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";

// global state
const allLabels = ["sheep", "dragon", "mona_lisa", "guitar", "pig",
             "tree", "clock"]
const targetLabels = ["clock", "guitar"];

const confidenceChart = new ConfidenceChart(allLabels, targetLabels)
const distanceIndicator = new DistanceIndicator(150, 0)
const drawingBoard = new DrawingBoard(distanceIndicator)
const inferencer = new Inferencer()

var inferenceMutex = false // true for locked, false for unlocked

drawingBoard.afterMouseEnd = async () => {
    const previewImageData = await drawingBoard.updatePreview()
    const imageDataUrl = drawingBoard.previewCanvas.toDataURL();
    const modelOutputs = await inferencer.serverInferImage(imageDataUrl)
    confidenceChart.update(modelOutputs)
}

drawingBoard.afterMouseMove = async () => {
    if (!inferenceMutex) {
        inferenceMutex = true

        const previewImageData = await drawingBoard.updatePreview()
        const modelOutputs = await inferencer.clientInferImage(previewImageData)
        confidenceChart.update(modelOutputs)
        inferenceMutex = false
    }
}

//TODO: window.onresize = () =>
