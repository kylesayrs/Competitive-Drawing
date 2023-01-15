/*
filename: turn_indicator.js
author: Kyle Sayers
details: Shows whose turn it is, how many turns are left, and who is the winner
*/

export class TurnIndicator {
    constructor(totalNumTurns, currentTurnTarget, debug=false) {
        this.totalNumTurns = totalNumTurns
        this.debug = debug

        this.turnsLeftIndicator = document.querySelector("#turns-left-indicator");
        this.playerTurnIndicator = document.querySelector("#player-turn-indicator");
        this.endTurnButton = document.querySelector("#endTurnButton");

        this.turnsLeft = totalNumTurns
        this.currentTurnTarget = currentTurnTarget
    }

    get turnsLeft() {
        return this._turnsLeft
    }

    set turnsLeft(newTurnsLeft) {
        this._turnsLeft = newTurnsLeft
        this.turnsLeftIndicator.innerHTML = newTurnsLeft
    }

    get currentTurnTarget() {
        return this._currentTurnTarget
    }

    set currentTurnTarget(newTurnTarget) {
        this._currentTurnTarget = newTurnTarget
        this.playerTurnIndicator.innerHTML = "Turn: " + newTurnTarget
    }

    update(turnsLeft, newTurnTarget) {
        this.turnsLeft = turnsLeft
        this.currentTurnTarget = newTurnTarget
    }

    showEndTurnButton() {
        this.endTurnButton.setAttribute("hidden", false)
    }

    hideEndTurnButton() {
        this.endTurnButton.setAttribute("hidden", true)
    }

    showWinner(winnerTarget) {
        this.turnsLeftIndicator.setAttribute("hidden", true)
        this.playerTurnIndicator.innerHTML = "Winner: " + winnerTarget
        this.hideEndTurnButton()
    }
}
