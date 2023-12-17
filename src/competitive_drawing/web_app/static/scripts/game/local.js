import { GameBase } from "/static/scripts/game/base.js";


export class LocalGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(1, gameConfig, debug)

        // Initialize components
        this.drawingBoard.enabled = true
        this.distanceIndicator.resetDistance()
        this.turnIndicator.showEndTurnButton()
    }


    onStartTurn(data) {
        super.onStartTurn(data)

        this.playerId = data["turn"]
    }


    onEndTurnButtonClick(_event) {
        super.onEndTurnButtonClick()

        this.distanceIndicator.resetDistance()
    }
}


export class SinglePlayerGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(1, gameConfig, debug)

        // Player variables
        this.playerId = null

        // Initialize components
        this.drawingBoard.enabled = true
        this.distanceIndicator.resetDistance()

        // Join local room
        this.socket.emit("join_room", {
            "roomId": this.roomId
        })

    }


    onStartTurn(data) {
        super.onStartTurn(data)

        this.playerId = data["turn"]
    }


    onEndTurnButtonClick(_event) {
        this.distanceIndicator.resetDistance()
        super.onEndTurnButtonClick()
    }
}
