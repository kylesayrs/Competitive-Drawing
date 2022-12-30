/*
filename: confidence_bar
author: Kyle Sayers
details: ConfidenceBar is used to control the confidence bar. It also does some
         data processing like normalization and softmax
*/
import { normalize, softmax } from "/static/scripts/helpers.js";

export class ConfidenceBar {
    constructor(allLabels, targetLabels=null, softmaxFactor=7) {
        this.allLabels = allLabels
        this._targetLabels = targetLabels != null ? targetLabels : allLabels
        this.softmaxFactor = softmaxFactor

        this.confidenceBar = document.querySelector("#confidence-bar");
        this.leftConfidence = document.querySelector("#left-confidence");
        this.rightConfidence = document.querySelector("#right-confidence");
        this.leftConfidenceLabel = document.querySelector("#left-confidence-label");
        this.rightConfidenceLabel = document.querySelector("#right-confidence-label");

        this.leftConfidenceLabel.innerHTML = this._targetLabels[0] || "Left label"
        this.rightConfidenceLabel.innerHTML = this._targetLabels[1] || "Right label"
    }

    get targetLabels() {
        return this._targetLabels
    }

    set targetLabels(targetLabels) {
        this._targetLabels = targetLabels
    }

    update(modelOutputs) {
        // normalize scores
        const modelOutputsNormalized = normalize(modelOutputs, 0, 1)

        // filter to target outputs
        // TODO use .filter()
        var filteredOutputs = []
        var filteredLabels = []
        for (let i = 0; i < this.allLabels.length; i++) {
            if (this._targetLabels.includes(this.allLabels[i])) {
                filteredOutputs.push(modelOutputsNormalized[i])
                filteredLabels.push(this.allLabels[i])
            }
        }

        // apply softmax
        const modelConfidences = softmax(filteredOutputs, this.softmaxFactor)

        // draw data
        const firstIndex = filteredLabels.indexOf(this._targetLabels[0])
        const secondIndex = filteredLabels.indexOf(this._targetLabels[1])
        this.leftConfidence.style.width = modelConfidences[firstIndex] * 100 + "%";
        this.rightConfidence.style.width = modelConfidences[secondIndex] * 100 + "%";
    }
}
