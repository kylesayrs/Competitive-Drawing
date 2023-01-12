import { GameBase } from "/static/scripts/game/base.js";


export class SinglePlayerGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(3, gameConfig, debug)

        // Player variables
        this.playerId = null
        this.humanId = null
        this.humanTargetIndex = null
        this.aiId = null
        this.aiTargetIndex = null
    }


    onStartGame(data) {
        super.onStartGame(data)
        console.log(data)

        // TODO: randomize
        this.humanId = Object.keys(data["targets"])[0]
        this.aiId = Object.keys(data["targets"])[1]

        // assign target index
        this.humanTargetIndex = data["targetIndices"][this.humanId]
        this.aiTargetIndex = data["targetIndices"][this.aiId]
    }


    async onStartTurn(data) {
        super.onStartTurn(data)

        this.playerId = data["turn"]
        if (data["turn"] == this.humanId) {
            console.log("user turn")
            this.drawingBoard.enabled = true
            this.distanceIndicator.resetDistance()
            if (this.debug) {
                this.distanceIndicator.mouseDistance = -2000
            }
        } else if (data["turn"] == this.aiId) {
            console.log("ai turn")
            this.drawingBoard.enabled = false
            this.distanceIndicator.resetDistance()

            // Await API request to get stroke data
            const strokeSamples = await this.serverInferAIStroke()
            console.log("strokeSamples")
            console.log(strokeSamples)

            // Simulate drawing the stroke on the drawingBoard
            await this.drawingBoard.replayStroke(strokeSamples, 3000)

            // End AI turn
            this.onEndTurnButtonClick()
            console.log("ended ai turn")
        } else {
            console.log("WARNING: Unknown turn " + data["turn"])
            console.log(
                "WARNING: Known ids are human: " + this.playerId + "and " +
                "ai: " + this.aiId
            )
        }
    }


    async serverInferAIStroke() {
        if (!this.inferencer) {
            console.log("ERROR: Inferencer not initialized yet!")
            return
        }

        await this.drawingBoard.updatePreview()
        const imageDataUrl = this.drawingBoard.getPreviewImageDataUrl()
        return this.inferencer.serverInferStroke(imageDataUrl, this.aiTargetIndex)
    }
}
