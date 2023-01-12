import { GameBase } from "/static/scripts/game/base.js";


export class SinglePlayerGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(3, gameConfig, debug)

        // Player variables
        this.playerId = null
        this.humanId = null
        this.aiId = null
    }


    onAssignPlayer(data) {
        // Assign human player the first player id
        if (this.humanId == null) {
            this.humanId = data["playerId"]
        } else {
            this.aiId = data["playerId"]
        }
    }


    async onStartTurn(data) {
        super.onStartTurn(data)

        if (this.humanId == data["turn"]) {
            console.log("user turn")
            this.playerId = this.humanId
            this.drawingBoard.enabled = true
            this.distanceIndicator.resetDistance()
        } else {
            console.log("ai turn")
            this.playerId = this.aiId
            this.drawingBoard.enabled = false
            this.distanceIndicator.emptyDistance()

            // TODO: Integrate with model service
            // await API request to get stroke data

            // Simulate drawing the stroke on the drawingBoard
            // drawingBoard.replay_stroke(keypoints)

            this.onEndTurnButtonClick()
            console.log("ended ai turn")
        }
    }
}
