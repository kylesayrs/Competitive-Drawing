import { GameBase } from "/static/scripts/game/base.js";


export class SinglePlayerGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(3, gameConfig, debug)

        // Player variables
        this.humanId = null
        this.humanTargetIndex = null
        this.aiId = null
        this.aiTargetIndex = null

        // custom socket
        this.socket.on("ai_stroke", this.onAIStroke.bind(this))
    }


    get aiInferenceMutex() {
        return JSON.parse(window.sessionStorage.getItem("aiInferenceMutex"))
    }


    set aiInferenceMutex(value) {
        window.sessionStorage.setItem("aiInferenceMutex", JSON.stringify(value))
    }


    onAssignPlayer(data) {}  // ignore player assignment messages


    onStartGame(data) {
        super.onStartGame(data)

        if (Object.keys(data["targets"]).includes(this.playerId)) {
            // resuming game
            console.log("resuming game " + this.playerId)
            this.humanId = this.playerId
            const humanIdIndex = Object.keys(data["targets"]).indexOf(this.humanId)
            this.aiId = Object.keys(data["targets"])[1 - humanIdIndex]
        } else {
            // new game
            // TODO: randomize
            console.log("new game")
            this.humanId = Object.keys(data["targets"])[0]
            this.aiId = Object.keys(data["targets"])[1]
            this.playerId = this.humanId
            this.aiInferenceMutex = false

            Toastify({
                text: "Your target is " + data["targets"][this.humanId],
                duration: 10000,
                className: "info",
                gravity: "toastify-top",
                style: {
                    background: "linear-gradient(to right, #2ec1cc, #0074d9)",
                }
            }).showToast();
        }

        // assign target index
        this.humanTargetIndex = data["targetIndices"][this.humanId]
        this.aiTargetIndex = data["targetIndices"][this.aiId]
    }


    async onStartTurn(data) {
        await super.onStartTurn(data)

        if (data["turn"] == this.humanId) {
            console.log("user turn")
            this.drawingBoard.enabled = true
            this.distanceIndicator.resetDistance()
            this.turnIndicator.showEndTurnButton()

        } else if (data["turn"] == this.aiId) {
            this.drawingBoard.enabled = false
            this.distanceIndicator.resetDistance()
            this.turnIndicator.hideEndTurnButton()
            
            if (!this.aiInferenceMutex) {
                this.aiInferenceMutex = true
            }

            Toastify({
                text: "AI is computing (~60 seconds)...",
                duration: 10000,
                className: "info",
                gravity: "toastify-top",
                style: {
                    background: "linear-gradient(to right, #00b09b, #96c93d)",
                }
            }).showToast();

        } else {
            console.log("WARNING: Unknown turn " + data["turn"])
            console.log(
                "WARNING: Known ids are human: " + this.humanId +
                " and ai: " + this.aiId
            )
        }
    }


    async onAIStroke(data) {
        // Simulate drawing the stroke on the drawingBoard
        await this.drawingBoard.replayStroke(data["strokeSamples"], 3000)

        // End AI turn
        const canvasDataUrl = this.drawingBoard.getCanvasImageDataUrl()
        await this.drawingBoard.updatePreview()
        const imageDataUrl = this.drawingBoard.getPreviewImageDataUrl()
        this.socket.emit("end_turn", {
            "roomId": this.roomId,
            "playerId": this.aiId,
            "canvas": canvasDataUrl,
            "preview": imageDataUrl,
        })
        this.aiInferenceMutex = false
    }


    onEndTurnButtonClick(_event) {
        // Only press if it is not the AI's turn
        if (!this.aiInferenceMutex) {
            super.onEndTurnButtonClick(_event);
        }
    }
}
