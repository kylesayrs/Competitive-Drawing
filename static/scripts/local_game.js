// libraries
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
// gameConfig from Flask

const targetLabels = ["spider", "pig"];

const confidenceChart = new ConfidenceChart(gameConfig.allLabels, targetLabels, gameConfig.softmaxFactor)
const distanceIndicator = new DistanceIndicator(140, 0)
const drawingBoard = new DrawingBoard(distanceIndicator, 500)
const inferencer = new Inferencer()

// global state
var inferenceMutex = false // true for locked, false for unlocked
var targetIndex = 4
drawingBoard.enabled = true

drawingBoard.afterMouseEnd = async () => {
    const previewImageData = await drawingBoard.updatePreview()
    const imageDataUrl = drawingBoard.previewCanvas.toDataURL();
    const { gradCamImage, modelOutputs } = await inferencer.serverInferImage(imageDataUrl, targetIndex)
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
