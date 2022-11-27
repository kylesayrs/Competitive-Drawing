/*
filename: distance_indicator.js
author  : Kyle Sayers
details : The DistanceIndicator is responsible for controlling and keeping track
          of drawing distances. It can be used as arguments to other classes for
          control and access to drawing distances.
*/

export class DistanceIndicator {
    constructor(mouseDistanceLimit, totalMouseDistance) {
        this.distanceBottom = document.getElementById("distance-bottom")
        this.buttonElement = document.getElementById("distanceIndicatorButton")

        this._mouseDistanceLimit = mouseDistanceLimit
        this._totalMouseDistance = totalMouseDistance

        this.buttonElement.onclick = (_event) => {
            this.totalMouseDistance = 0;
            this.afterOnClick();
        }

        this.update()
    }

    get totalMouseDistance() {
        return this._totalMouseDistance
    }

    set totalMouseDistance(value) {
        this._totalMouseDistance = value
        this.update()
    }

    get mouseDistanceLimit() {
        return this._mouseDistanceLimit
    }

    set mouseDistanceLimit(value) {
        this._mouseDistanceLimit = value
        this.update()
    }

    afterOnClick() {}

    update() {
        var percentLeft = 100 * this._totalMouseDistance / this._mouseDistanceLimit
        this.distanceBottom.style.height = (100 - percentLeft)  + "%";
    }
}
