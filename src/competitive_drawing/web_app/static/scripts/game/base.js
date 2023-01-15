// Imports
import { ConfidenceBar } from "/static/scripts/components/confidence_bar.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { TurnIndicator } from "/static/scripts/components/turn_indicator.js";
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
        this.socket.on("end_game", this.onEndGame.bind(this))

        // Components
        this.distanceIndicator = new DistanceIndicator(this.gameConfig.distancePerTurn)
        this.drawingBoard = new DrawingBoard(this.distanceIndicator, this.gameConfig)
        this.confidenceBar = new ConfidenceBar(this.gameConfig.softmaxFactor, this.debug)
        this.turnIndicator = new TurnIndicator(this.debug)
        //this.drawingBoard.afterMouseEnd = this.serverInferImage.bind(this)
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
        return window.sessionStorage.getItem("playerId")
    }


    set playerId(value) {
        window.sessionStorage.setItem("playerId", value)
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

        // initialize canvas and confidence bar
        const canvasImageData = imageToImageData(
            data["canvas"],
            this.drawingBoard.canvasSize,
            this.drawingBoard.canvasSize
        )
        this.drawingBoard.putPreviewImageData(canvasImageData, true)
        this.confidenceBar.targetLabels = Object.values(data["targets"])

        // initialize preview and preview (not confidence bar)
        await this.drawingBoard.updatePreview()
        //this.serverInferImage()
    }


    async onStartTurn(data) {
        if (this.debug) {
            console.log("start_turn")
            console.log(data)
        }

        // update canvas and preview
        const canvasImageData = imageToImageData(
            data["canvas"],
            this.drawingBoard.canvasSize,
            this.drawingBoard.canvasSize
        )
        this.drawingBoard.putCanvasImageData(canvasImageData, true)
        await this.drawingBoard.updatePreview()

        // update turn indicator
        this.turnIndicator.update(data["turnsLeft"], data["target"])

        // update confidences and preview image
        //this.clientInferImage()
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

    async onEndTurnButtonClick(_event) {
        this.serverInferImage()

        const canvasDataUrl = this.drawingBoard.getCanvasImageDataUrl()
        await this.drawingBoard.updatePreview()
        const imageDataUrl = this.drawingBoard.getPreviewImageDataUrl()
        this.socket.emit("end_turn", {
            "game_type": this.gameType,
            "roomId": this.roomId,
            "playerId": this.playerId,
            "canvas": canvasDataUrl,
            "preview": imageDataUrl,
            //replay data
        })
    }


    onEndGame(data) {
        if (this.debug) {
            console.log("end_game")
            console.log(data)
        }

        this.turnIndicator.showWinner(data["winnerTarget"])
        this.drawingBoard.enabled = false
        this.distanceIndicator.emptyDistance()
    }
}
