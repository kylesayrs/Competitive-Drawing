export class DistanceIndicator {
    constructor(mouseDistanceLimit, totalMouseDistance) {
        this.indicatorElement = document.getElementById("distanceIndicator")
        this.buttonElement = document.getElementById("distanceIndicatorButton")

        this._mouseDistanceLimit = mouseDistanceLimit
        this._totalMouseDistance = totalMouseDistance

        this.buttonElement.onclick = (_event) => {
            this.totalMouseDistance = 0;
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

    update(value) {
        this.indicatorElement.innerHTML = "Distance remaining: " + Math.round(this._mouseDistanceLimit - this._totalMouseDistance).toString();
    }
}
