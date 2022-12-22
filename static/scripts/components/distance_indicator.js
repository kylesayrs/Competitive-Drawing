/*
filename: distance_indicator.js
author  : Kyle Sayers
details : The DistanceIndicator is responsible for controlling and keeping track
          of drawing distances. It can be used as arguments to other classes for
          control and access to drawing distances.
*/

export class DistanceIndicator {
    constructor(mouseDistanceLimit, mouseDistance=0) {
        this.distanceBottom = document.querySelector("#distance-bottom")
        this.buttonElement = document.querySelector("#distanceIndicatorButton")

        this._mouseDistanceLimit = mouseDistanceLimit
        this._mouseDistance = mouseDistance

        this.buttonElement.onclick = (_event) => {
            this.onButtonClick(_event)
        }

        this.update()
    }

    get mouseDistance() {
        return this._mouseDistance
    }

    set mouseDistance(value) {
        this._mouseDistance = value
        this.update()
    }

    get mouseDistanceLimit() {
        return this._mouseDistanceLimit
    }

    set mouseDistanceLimit(value) {
        this._mouseDistanceLimit = value
        this.update()
    }

    get distanceRemaining() {
        return this._mouseDistanceLimit - this._mouseDistance
    }

    resetDistance() {
        this.mouseDistance = 0
    }

    emptyDistance() {
        this.mouseDistance = this._mouseDistanceLimit
    }

    onButtonClick(_event) {
        this.resetDistance()
    }

    update() {
        var percentLeft = 100 * this._mouseDistance / this._mouseDistanceLimit
        this.distanceBottom.style.height = (100 - percentLeft)  + "%";
    }
}
