import { GameBase } from "/static/scripts/game/base.js";


export class LocalGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(1, gameConfig, debug)
        
        // Initialize components
        this.drawingBoard.enabled = true
        this.distanceIndicator.resetDistance()
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
            "room_id": this.roomId,
            "game_type": this.gameType,
        })

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