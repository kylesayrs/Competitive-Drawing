// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
// allLabels from flask

// components
const drawingBoard = new DrawingBoard()
const confidenceChart = new ConfidenceChart(allLabels)
const inferencer = new Inferencer(drawingBoard)

// game state
var inferenceMutex = false

function gradCamImageToImageData(gradCamImage) {
    var imageDataBuffer = []
    for (let y = 0; y < gradCamImage.length; y++) {
        for (let x = 0; x < gradCamImage[y].length; x++) {
            for (let c = 0; c < gradCamImage[y][x].length; c++) {
                imageDataBuffer.push(gradCamImage[y][x][c])
            }
            imageDataBuffer.push(255)
        }
    }

    const imageData = new ImageData(
        new Uint8ClampedArray(imageDataBuffer),
        28,
        28,
    )

    return imageData;
}

// distanceIndicator.onEnd = () => serverInferImage
drawingBoard.afterMouseEnd = async () => {
    const previewImageData = await drawingBoard.updatePreview()
    const imageDataUrl = drawingBoard.previewCanvas.toDataURL();
    const { modelOutputs, gradCamImage } = await inferencer.serverInferImage(imageDataUrl)
    confidenceChart.update(modelOutputs)
    console.log(gradCamImage)
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
