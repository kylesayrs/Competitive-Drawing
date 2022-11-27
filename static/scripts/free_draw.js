// libraries
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { imageToImageData } from "/static/scripts/helpers.js";

// components
const drawingBoard = new DrawingBoard(null, 500)
const confidenceChart = new ConfidenceChart(gameConfig.allLabels, null, gameConfig.softmaxFactor)
const inferencer = new Inferencer(drawingBoard)

// game state
var inferenceMutex = false
var targetIndex = 0
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

        await drawingBoard.updatePreview()
        const previewImageData = drawingBoard.getPreviewImageData()
        const modelOutputs = await inferencer.clientInferImage(previewImageData)
        confidenceChart.update(modelOutputs)
        inferenceMutex = false
    }
}

//TODO: window.onresize = () =>
