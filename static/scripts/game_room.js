// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { PlayerGameState } from "/static/scripts/components/player_game_state.js";
// gameConfig from Flask

const confidenceChart = new ConfidenceChart(gameConfig.allLabels, null, gameConfig.softmaxFactor)
const distanceIndicator = new DistanceIndicator(80, 0)
const drawingBoard = new DrawingBoard(distanceIndicator)
const inferencer = new Inferencer()

// global state
const urlSearchParams = new URLSearchParams(window.location.search);
const urlParams = Object.fromEntries(urlSearchParams.entries());
var roomId = urlParams["room_id"]
var inferenceMutex = false // true for locked, false for unlocked
var playerGameState = new PlayerGameState()
drawingBoard.enabled = false

// socketio
var socket = io()

socket.emit("join_room", {
    "room_id": roomId
})

socket.on("assign_player", (data) => {
    playerGameState.playerId = data["playerId"]
})

socket.on("start_game", (data) => {
    // TODO: use find and better stuff
    var targetLabels = []
    for (const playerId in data["targets"]) {
        if (playerId == playerGameState.playerId) {
            playerGameState.target = data["targets"][playerId]
        }
        targetLabels.push(data["targets"][playerId])
    }

    confidenceChart.targetLabels = targetLabels
})

socket.on("start_turn", (data) => {
    if (data["turn"] == playerGameState.playerId) {
        playerGameState.myTurn = true
        drawingBoard.enabled = true
    } else {
        drawingBoard.enabled = false
    }
})

// distanceIndicator.onEnd = () => serverInferImage
drawingBoard.afterMouseEnd = async () => {
    if (playerGameState.myTurn) {
        const previewImageData = await drawingBoard.updatePreview()
        const imageDataUrl = drawingBoard.previewCanvas.toDataURL();
        const { gradCamImage, modelOutputs } = await inferencer.serverInferImage(imageDataUrl, playerGameState.target)
        confidenceChart.update(modelOutputs)
    }
}

drawingBoard.afterMouseMove = async () => {
    if (!inferenceMutex && playerGameState.myTurn) {
        inferenceMutex = true

        const previewImageData = await drawingBoard.updatePreview()
        const modelOutputs = await inferencer.clientInferImage(previewImageData)
        confidenceChart.update(modelOutputs)
        inferenceMutex = false
    }
}

distanceIndicator.afterOnClick = () => {
    socket.emit("end_turn", {
        "roomId": roomId,
        "playerId": playerGameState.playerId
    })
    drawingBoard.enabled = false
}

//TODO: window.onresize = () =>
