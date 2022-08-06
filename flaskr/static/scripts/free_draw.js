// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { gradCamImageToImageData } from "/static/scripts/helpers.js";
// allLabels from flask

// components
const drawingBoard = new DrawingBoard()
const confidenceChart = new ConfidenceChart(allLabels)
const inferencer = new Inferencer(drawingBoard)

// game state
var inferenceMutex = false
var targetIndex = 0

// distanceIndicator.onEnd = () => serverInferImage
drawingBoard.afterMouseEnd = async () => {
    const previewImageData = await drawingBoard.updatePreview()
    const imageDataUrl = drawingBoard.previewCanvas.toDataURL();
    const { modelOutputs, gradCamImage } = await inferencer.serverInferImage(imageDataUrl, targetIndex)
    confidenceChart.update(modelOutputs)
    const gradImageData = gradCamImageToImageData(gradCamImage)
    drawingBoard.previewCanvasContext.putImageData(gradImageData, 0, 0)
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
