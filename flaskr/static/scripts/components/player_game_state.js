export class PlayerGameState {
    constructor() {
        this.inferenceMutex = false // true for locked, false for unlocked
        this.playerId = null
        this.playerTarget = 0
    }
}
