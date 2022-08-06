// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { PlayerGameState } from "/static/scripts/components/player_game_state.js";
var socket = io()
// allLabels from flask

const confidenceChart = new ConfidenceChart(allLabels)
const distanceIndicator = new DistanceIndicator(80, 0)
const drawingBoard = new DrawingBoard(distanceIndicator)
const inferencer = new Inferencer()

// global state
var playerGameState = new PlayerGameState()

// socketio
const urlSearchParams = new URLSearchParams(window.location.search);
const params = Object.fromEntries(urlSearchParams.entries());
socket.emit("join_room", params)

socket.on("assign_player", function(data) {
    console.log("assign_player")
    console.log(data)
    playerGameState.playerId = data["playerId"]
})

socket.on("start_game", function(data) {
    console.log("start_game")
    console.log(data)

    // TODO: use find and better stuff
    var targetLabels = []
    for (const playerId in data["targets"]) {
        if (playerId == playerGameState.playerId) {
            playerGameState.playerTarget = data["targets"][playerId]
        }
        targetLabels.push(data["targets"][playerId])
    }

    confidenceChart.targetLabels = targetLabels
})

// distanceIndicator.onEnd = () => serverInferImage
drawingBoard.afterMouseEnd = async () => {
    const previewImageData = await drawingBoard.updatePreview()
    const imageDataUrl = drawingBoard.previewCanvas.toDataURL();
    const { gradCamImage, modelOutputs } = await inferencer.serverInferImage(imageDataUrl, playerGameState.playerTarget)
    confidenceChart.update(modelOutputs)
}

drawingBoard.afterMouseMove = async () => {
    if (!playerGameState.inferenceMutex) {
        playerGameState.inferenceMutex = true

        const previewImageData = await drawingBoard.updatePreview()
        const modelOutputs = await inferencer.clientInferImage(previewImageData)
        confidenceChart.update(modelOutputs)
        playerGameState.inferenceMutex = false
    }
}

//TODO: window.onresize = () =>
