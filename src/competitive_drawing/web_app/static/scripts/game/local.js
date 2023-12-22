import { GameBase } from "/static/scripts/game/base.js";


export class LocalGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(1, gameConfig, debug)

        // Initialize components
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
