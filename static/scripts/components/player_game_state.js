export class PlayerGameState {
    constructor(gameConfig) {
        this._gameConfig = gameConfig
        this.playerId = null
        this._target_index = -1
        this.myTurn = false
    }

    get target() {
        return this._gameConfig.allLabels[this._target_index]
    }

    set target(targetValue) {
        this._target_index = this._gameConfig.allLabels.indexOf(targetValue)
    }

    get targetIndex() {
        return this._target_index
    }

    set targetIndex(newIndex) {
        this._target_index = newIndex
    }
}
