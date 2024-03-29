/*
filename: confidence_bar
author: Kyle Sayers
details: ConfidenceBar is used to control the confidence bar. It also does some
         data processing like normalization and softmax
*/
import { normalize, softmax } from "/static/scripts/helpers.js";

export class ConfidenceBar {
    constructor(softmaxFactor=1.0, debug=false) {
        this.softmaxFactor = softmaxFactor
        this.debug = debug

        this.confidenceBar = document.querySelector("#confidence-bar");
        this.leftConfidence = document.querySelector("#left-confidence");
        this.leftConfidenceLabel = document.querySelector("#left-confidence-label");
        this.rightConfidenceLabel = document.querySelector("#right-confidence-label");

        // initialize mock labels
        this.leftConfidenceLabel.innerHTML = ""
        this.rightConfidenceLabel.innerHTML = ""
        this._targetLabels = ["", ""]

        this.leftConfidence.style.width = "50%";
    }

    get targetLabels() {
        return this._targetLabels
    }

    set targetLabels(targetLabels) {
        this._targetLabels = targetLabels

        this.leftConfidenceLabel.innerHTML = this._targetLabels[0]
        this.rightConfidenceLabel.innerHTML = this._targetLabels[1]
    }

    update(modelOutputs) {
        // filter to target outputs
        // TODO use .filter() or some sort method
        var filteredOutputs = []
        var filteredLabels = []
        for (let i = 0; i < this.targetLabels.length; i++) {
            if (this._targetLabels.includes(this.targetLabels[i])) {
                filteredOutputs.push(modelOutputs[i])
                filteredLabels.push(this.targetLabels[i])
            }
        }

        function normalize(vector) {
            const magnitude = Math.sqrt(Math.pow(vector[0], 2) + Math.pow(vector[1], 2))
            return [vector[0] / magnitude, vector[1] / magnitude]
        }

        // apply softmax
        const modelConfidences = softmax(filteredOutputs, this.softmaxFactor)
        if (this.debug) {
            console.log(filteredOutputs)
            console.log(modelConfidences)
            const magnitude = Math.sqrt(Math.pow(filteredOutputs[0], 2) + Math.pow(filteredOutputs[1], 2))
            console.log(magnitude)
        }

        // draw data
        const firstIndex = filteredLabels.indexOf(this._targetLabels[0])
        this.leftConfidence.style.width = modelConfidences[firstIndex] * 100 + "%";
    }
}
