// libraries
import { ConfidenceBar } from "/static/scripts/components/confidence_bar.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { PlayerGameState } from "/static/scripts/components/player_game_state.js";
import { imageToImageData } from "/static/scripts/helpers.js";

const confidenceBar = new ConfidenceBar(gameConfig.allLabels, null, gameConfig.softmaxFactor)
const distanceIndicator = new DistanceIndicator(gameConfig.distancePerTurn)
const drawingBoard = new DrawingBoard(distanceIndicator, gameConfig.canvasSize)
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

socket.on("start_game", async (data) => {
    // TODO: use find
    var targetLabels = []
    for (const playerId in data["targets"]) {
        if (playerId == playerGameState.playerId) {
            playerGameState.target = data["targets"][playerId]
        }
        targetLabels.push(data["targets"][playerId])
    }

    // update canvas
    const canvasImageData = imageToImageData(
        data["canvas"],
        drawingBoard.canvasSize,
        drawingBoard.canvasSize
    )
    drawingBoard.putPreviewImageData(canvasImageData, true)

    confidenceBar.targetLabels = targetLabels

    // initalize bar
    await drawingBoard.updatePreview()
    const previewImageData = drawingBoard.getPreviewImageData()
    const modelOutputs = await inferencer.clientInferImage(previewImageData)
    confidenceBar.update(modelOutputs)
})

socket.on("start_turn", (data) => {
    console.log("start_turn")
    console.log(data)
    // update canvas
    console.log(drawingBoard.canvasSize)
    const canvasImageData = imageToImageData(
        data["canvas"],
        drawingBoard.canvasSize,
        drawingBoard.canvasSize
    )
    drawingBoard.putCanvasImageData(canvasImageData, true)

    if (data["turn"] == playerGameState.playerId) {
        playerGameState.myTurn = true
        drawingBoard.enabled = true
        distanceIndicator.resetDistance()
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
        confidenceBar.update(modelOutputs)
    }
}

drawingBoard.afterMouseMove = async () => {
    if (!inferenceMutex && playerGameState.myTurn) {
        inferenceMutex = true

        await drawingBoard.updatePreview()
        const previewImageData = drawingBoard.getPreviewImageData()
        const modelOutputs = await inferencer.clientInferImage(previewImageData)
        confidenceBar.update(modelOutputs)

        inferenceMutex = false
    }
}

distanceIndicator.onButtonClick = (_event) => {
    const imageDataUrl = drawingBoard.getCanvasImageDataUrl()

    socket.emit("end_turn", {
        "roomId": roomId,
        "playerId": playerGameState.playerId,
        "canvas": imageDataUrl
        //replay data
    })

    drawingBoard.enabled = false
    distanceIndicator.emptyDistance()
}

// initalize distance indicator
distanceIndicator.emptyDistance()
