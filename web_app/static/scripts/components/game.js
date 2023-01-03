// Imports
import { ConfidenceBar } from "/static/scripts/components/confidence_bar.js";
import { DistanceIndicator } from "/static/scripts/components/distance_indicator.js";
import { DrawingBoard } from "/static/scripts/components/drawing_board.js";
import { Inferencer } from "/static/scripts/components/inference.js";
import { getRoomIdFromUrl, imageToImageData } from "/static/scripts/helpers.js";

class MultiplayerGameBase {
    constructor(gameConfig, debug=false) {
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

        // Inference
        this.inferenceMutex = false  // true for locked, false for unlocked
        this.inferencer = null
    }


    onAssignPlayer(data) {}


    async onStartGame(data) {
        if (self.debug) {
            console.log("onStartGame")
            console.log(data)
        }

        // Initialize inferencer
        this.inferencer = new Inferencer()
        await this.inferencer.loadModel(data["onnxUrl"])
        console.log(this.inferencer)

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
        if (self.debug) {
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

        this.inferenceMutex = true

        await this.drawingBoard.updatePreview()
        const imageDataUrl = this.drawingBoard.getPreviewImageDataUrl()
        const modelOutputs = await this.inferencer.serverInferImage(imageDataUrl, this.playerTargetIndex)
        this.confidenceBar.update(modelOutputs)

        this.inferenceMutex = false
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

        this.socket.emit("end_turn", {
            "roomId": this.roomId,
            "playerId": this.playerId,
            "canvas": imageDataUrl
            //replay data
        })
    }
}

export class OnlineGame extends MultiplayerGameBase {
    constructor(gameConfig, debug=false) {
        super(gameConfig, debug)

        // Player specific variables
        this.playerId = null
        this.playerTargetIndex = null
        this.myTurn = false

        // Hook with drawing board
        this.drawingBoard.afterMouseEnd = async () => {
            if (this.myTurn) {
                this.serverInferImage()
            }
        }

        // Initialize components
        this.drawingBoard.enabled = false
        this.distanceIndicator.emptyDistance()

        // Join online room
        this.socket.emit("join_room", {
            "room_id": this.roomId,
            "room_type": 2
        })
    }

    onAssignPlayer(data) {
        this.playerId = data["playerId"]
    }

    onStartGame(data) {
        super.onStartGame(data)

        // assign target index
        this.playerTargetIndex = data["targets"][this.playerId]
    }


    async onStartTurn(data) {
        super.onStartTurn(data)

        if (data["turn"] == playerGameState.playerId) {
            this.myTurn = true
            this.drawingBoard.enabled = true
            this.distanceIndicator.resetDistance()
        } else {
            this.myTurn = false
            this.drawingBoard.enabled = false
        }
    }


    onEndTurnButtonClick(_event) {
        super.onEndTurnButtonClick(_event)

        this.drawingBoard.enabled = false
        this.distanceIndicator.emptyDistance()
    }
}


export class LocalGame extends MultiplayerGameBase {
    constructor(gameConfig, debug=false) {
        super(gameConfig, debug)

        // Initialize components
        this.drawingBoard.enabled = true
        this.distanceIndicator.resetDistance()

        // Join local room
        this.socket.emit("join_room", {
            "room_id": this.roomId,
            "game_type": 1,
        })
    }
}