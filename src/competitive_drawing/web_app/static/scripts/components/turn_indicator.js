/*
filename: turn_indicator.js
author: Kyle Sayers
details: Shows whose turn it is, how many turns are left, and who is the winner
*/

export class TurnIndicator {
    constructor(debug=false) {
        this.debug = debug

        this.turnsLeftIndicator = document.querySelector("#turns-left-indicator");
        this.playerTurnIndicator = document.querySelector("#player-turn-indicator");
        this.endTurnButton = document.querySelector("#endTurnButton");
    }

    get turnsLeft() {
        return this._turnsLeft
    }

    set turnsLeft(newTurnsLeft) {
        this._turnsLeft = newTurnsLeft
        this.turnsLeftIndicator.innerHTML = "Turns left: " + newTurnsLeft
    }

    get currentTurnTarget() {
        return this._currentTurnTarget
    }

    set currentTurnTarget(newTurnTarget) {
        this._currentTurnTarget = newTurnTarget
        this.playerTurnIndicator.innerHTML = newTurnTarget + "'s turn"
    }

    update(turnsLeft, newTurnTarget) {
        this.turnsLeft = turnsLeft
        this.currentTurnTarget = newTurnTarget
    }

    showEndTurnButton() {
        this.endTurnButton.removeAttribute("hidden")
    }

    hideEndTurnButton() {
        this.endTurnButton.setAttribute("hidden", true)
    }

    showWinner(winnerTarget) {
        this.turnsLeftIndicator.setAttribute("hidden", true)
        this.playerTurnIndicator.innerHTML = "Winner: " + winnerTarget + "!"
        this.hideEndTurnButton()
    }
}
