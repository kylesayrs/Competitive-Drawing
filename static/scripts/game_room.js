// libraries
import { ConfidenceChart } from "/static/scripts/components/confidence_chart.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { PlayerGameState } from "/static/scripts/components/player_game_state.js";
import { imageToImageData } from "/static/scripts/helpers.js";
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
var playerGameState = new PlayerGameState(gameConfig)
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
    console.log("start_game")
    console.log(data)
    var targetLabels = []
    for (const playerId in data["targets"]) {
        if (playerId == playerGameState.playerId) {
            playerGameState.target = data["targets"][playerId]
        }
        targetLabels.push(data["targets"][playerId])
    }

    // update canvas
    const canvasImageData = imageToImageData(data["canvas"], 500, 500)
    drawingBoard.putPreviewImageData(canvasImageData, true)

    confidenceChart.targetLabels = targetLabels
})

socket.on("start_turn", (data) => {
    console.log("start_turn")
    console.log(data)

    // update canvas
    const canvasImageData = imageToImageData(data["canvas"], 500, 500)
    drawingBoard.putCanvasImageData(canvasImageData, true)

    if (data["turn"] == playerGameState.playerId) {
        playerGameState.myTurn = true
        drawingBoard.enabled = true
    } else {
        playerGameState.myTurn = false
        drawingBoard.enabled = false
    }
})

// distanceIndicator.onEnd = () => serverInferImage
drawingBoard.afterMouseEnd = async () => {
    if (playerGameState.myTurn) {
        await drawingBoard.updatePreview()
        const imageDataUrl = drawingBoard.getPreviewImageDataUrl()
        const { gradCamImage, modelOutputs } = await inferencer.serverInferImage(imageDataUrl, playerGameState.targetIndex)
        confidenceChart.update(modelOutputs)
    }
}

drawingBoard.afterMouseMove = async () => {
    if (!inferenceMutex && playerGameState.myTurn) {
        inferenceMutex = true

        await drawingBoard.updatePreview()
        const previewImageData = drawingBoard.getPreviewImageData()
        const modelOutputs = await inferencer.clientInferImage(previewImageData)
        confidenceChart.update(modelOutputs)

        inferenceMutex = false
    }
}

distanceIndicator.afterOnClick = () => {
    const imageDataUrl = drawingBoard.getCanvasImageDataUrl()

    socket.emit("end_turn", {
        "roomId": roomId,
        "playerId": playerGameState.playerId,
        "canvas": imageDataUrl
        //replay data
    })
    drawingBoard.enabled = false
}

//TODO: window.onresize = () =>
