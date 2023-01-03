/*
filename: confidence_bar
author: Kyle Sayers
details: ConfidenceBar is used to control the confidence bar. It also does some
         data processing like normalization and softmax
*/
import { normalize, softmax } from "/static/scripts/helpers.js";

export class ConfidenceBar {
    constructor(softmaxFactor=7) {
        this.softmaxFactor = softmaxFactor

        this.confidenceBar = document.querySelector("#confidence-bar");
        this.leftConfidence = document.querySelector("#left-confidence");
        this.rightConfidence = document.querySelector("#right-confidence");
        this.leftConfidenceLabel = document.querySelector("#left-confidence-label");
        this.rightConfidenceLabel = document.querySelector("#right-confidence-label");
    }

    get targetLabels() {
        return this._targetLabels
    }

    set targetLabels(targetLabels) {
        this._targetLabels = targetLabels

        this.leftConfidenceLabel.innerHTML = this._targetLabels[0] || "Left label"
        this.rightConfidenceLabel.innerHTML = this._targetLabels[1] || "Right label"
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

        // apply softmax
        console.log(filteredOutputs)
        const modelConfidences = softmax(filteredOutputs, this.softmaxFactor)
        console.log(modelConfidences)

        // draw data
        const firstIndex = filteredLabels.indexOf(this._targetLabels[0])
        const secondIndex = filteredLabels.indexOf(this._targetLabels[1])
        this.leftConfidence.style.width = modelConfidences[firstIndex] * 100 + "%";
        this.rightConfidence.style.width = modelConfidences[secondIndex] * 100 + "%";
    }
}
