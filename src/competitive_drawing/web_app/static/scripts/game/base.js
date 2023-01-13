// Imports
import { ConfidenceBar } from "/static/scripts/components/confidence_bar.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { getRoomIdFromUrl, imageToImageData } from "/static/scripts/helpers.js";

export class GameBase {
    constructor(gameType, gameConfig, debug=false) {
        this.gameType = gameType
        this.gameConfig = gameConfig
        this.debug = debug

        // Sockets
        this.socket = io()
        this.roomId = getRoomIdFromUrl()
        this.socket.on("assign_player", this.onAssignPlayer.bind(this))
        this.socket.on("start_game", this.onStartGame.bind(this))
        this.socket.on("start_turn", this.onStartTurn.bind(this))

        // Components
        this.distanceIndicator = new DistanceIndicator(this.gameConfig.distancePerTurn)
        this.drawingBoard = new DrawingBoard(this.distanceIndicator, this.gameConfig)
        this.confidenceBar = new ConfidenceBar(this.gameConfig.softmaxFactor)
        this.drawingBoard.afterMouseEnd = this.serverInferImage.bind(this)
        this.drawingBoard.afterMouseMove = this.clientInferImage.bind(this)
        this.distanceIndicator.onButtonClick = this.onEndTurnButtonClick.bind(this)

        // Inference
        this.inferenceMutex = false  // true for locked, false for unlocked
        this.inferencer = null

        // Join room
        this.socket.emit("join_room", {
            "room_id": this.roomId,
            "game_type": this.gameType,
            "cachedPlayerId": this.playerId,
        })
    }


    get playerId() {
        return window.localStorage.getItem("playerId")
    }


    set playerId(value) {
        window.localStorage.setItem("playerId", value)
    }


    onAssignPlayer(data) {
        if (self.debug) {
            console.log("onAssignPlayer")
            console.log(data)
        }
        this.playerId = data["playerId"]
    }


    async onStartGame(data) {
        if (self.debug) {
            console.log("onStartGame")
            console.log(data)
        }

        // Initialize inferencer
        this.inferencer = new Inferencer(this.gameConfig, data["targets"])
        await this.inferencer.loadModel(data["onnxUrl"])

        // update canvas and confidence bar
        const canvasImageData = imageToImageData(
            data["canvas"],
            this.drawingBoard.canvasSize,
            this.drawingBoard.canvasSize
        )
        this.drawingBoard.putPreviewImageData(canvasImageData, true)
        this.confidenceBar.targetLabels = Object.values(data["targets"])

        // initialize bar
        await this.drawingBoard.updatePreview()
        const previewImageData = this.drawingBoard.getPreviewImageData()
        const modelOutputs = await this.inferencer.clientInferImage(previewImageData)
        this.confidenceBar.update(modelOutputs)
    }


    async onStartTurn(data) {
        if (this.debug) {
            console.log("start_turn")
            console.log(data)
        }

        // update canvas
        const canvasImageData = imageToImageData(
            data["canvas"],
            this.drawingBoard.canvasSize,
            this.drawingBoard.canvasSize
        )
        this.drawingBoard.putCanvasImageData(canvasImageData, true)

        // update confidences and preview image
        this.clientInferImage()
    }


    async serverInferImage() {
        if (!this.inferencer) {
            console.log("ERROR: Inferencer not initialized yet!")
            return
        }

        if (!this.inferenceMutex) {
            this.inferenceMutex = true

            await this.drawingBoard.updatePreview()
            const imageDataUrl = this.drawingBoard.getPreviewImageDataUrl()
            const modelOutputs = await this.inferencer.serverInferImage(imageDataUrl, this.playerTargetIndex)
            this.confidenceBar.update(modelOutputs)

            this.inferenceMutex = false
        }
    }


    async clientInferImage() {
        if (!this.inferencer) {
            console.log("ERROR: Inferencer not initialized yet!")
            return
        }

        if (!this.inferenceMutex) {
            this.inferenceMutex = true

            await this.drawingBoard.updatePreview()
            const previewImageData = this.drawingBoard.getPreviewImageData()
            const modelOutputs = await this.inferencer.clientInferImage(previewImageData)
            this.confidenceBar.update(modelOutputs)

            this.inferenceMutex = false
        }
    }

    onEndTurnButtonClick(_event) {
        const imageDataUrl = this.drawingBoard.getCanvasImageDataUrl()

        console.log("onEndTurnButtonClick")
        console.log(this.playerId)
        this.socket.emit("end_turn", {
            "game_type": this.gameType,
            "roomId": this.roomId,
            "playerId": this.playerId,
            "canvas": imageDataUrl,
            //replay data
        })
    }
}
