/*
filename: confidence_chart
author: Kyle Sayers
details: ConfidenceChart is used to control the confidence chart. It also does some
         data processing like normalization and softmax
*/

function normalize(arr, minNorm=0, maxNorm=1) {
    const minimumValue = Math.min.apply(Math, arr)
    arr = arr.map((value) => value - minimumValue, minNorm)
    const maxValue = Math.max.apply(Math, arr)
    const ratio = maxValue * maxNorm

    for (let i = 0; i < arr.length; i++ ) {
        arr[i] /= ratio;
    }
    return arr
}

function softmax(arr, factor=1) {
    const exponents = arr.map((value) => Math.exp(value * factor))
    const total = exponents.reduce((a, b) => a + b, 0);
    return exponents.map((exp) => exp / total);
}

export class ConfidenceChart {
    constructor(allLabels, targetLabels=null, softmaxFactor=7) {
        this.allLabels = allLabels
        this.targetLabels = targetLabels != null ? targetLabels : allLabels
        this.softmaxFactor = softmaxFactor

        const confidenceChartMargin = {"top": 20, "right": 30, "bottom": 40, "left": 90}
        const confidenceChartWidth = 460 - confidenceChartMargin.left - confidenceChartMargin.right
        const confidenceChartHeight = 400 - confidenceChartMargin.top - confidenceChartMargin.bottom;
        this.confidenceChartSvg = d3.select("#confidenceChart")
            .append("svg")
                .attr("width", confidenceChartWidth + confidenceChartMargin.left + confidenceChartMargin.right)
                .attr("height", confidenceChartHeight + confidenceChartMargin.top + confidenceChartMargin.bottom)
            .append("g")
                .attr("transform", "translate(" + confidenceChartMargin.left + "," + confidenceChartMargin.top + ")");

        this.x_scale = d3.scaleLinear()
            .domain([0, 1])
            .range([0, confidenceChartWidth]);

        this.confidenceChartSvg.append("g")
            .attr("transform", "translate(0," + confidenceChartHeight + ")")
            .call(d3.axisBottom(this.x_scale))
            .selectAll("text")
                .attr("transform", "translate(-10,0)rotate(-45)")
                .style("text-anchor", "end");

        this.y_scale = d3.scaleBand()
            .range([0, confidenceChartHeight])
            .domain(this.targetLabels)
            .padding(.1);

        this.confidenceChartSvg.append("g")
            .call(d3.axisLeft(this.y_scale))
    }

    update(modelOutputs) {
        // normalize scores
        const modelOutputsNormalized = normalize(modelOutputs.output.data, 0, 1)

        // filter to target outputs
        // TODO use .filter()
        var filteredOutputs = []
        var filteredLabels = []
        for (let i = 0; i < this.allLabels.length; i++) {
            if (this.targetLabels.includes(this.allLabels[i])) {
                filteredOutputs.push(modelOutputsNormalized[i])
                filteredLabels.push(this.allLabels[i])
            }
        }

        // apply softmax
        const modelConfidences = softmax(filteredOutputs, this.softmaxFactor)

        // build data
        var chartData = []
        for (let i = 0; i < this.targetLabels.length; i++) {
            chartData.push({"label": filteredLabels[i], "value": modelConfidences[i]})
        }

        const dataRects = document.querySelectorAll(".dataRect");
        if (dataRects.length > 0) {
            for (const dataRect of dataRects) {
                dataRect.parentNode.removeChild(dataRect)
            }
        }

        this.confidenceChartSvg.selectAll()
            .data(chartData)
            .enter()
            .append("rect")
            .attr("class", "dataRect")
            .attr("x", this.x_scale(0) )
            .attr("y", (d) => this.y_scale(d.label))
            .attr("width", (d) => this.x_scale(d.value))
            .attr("height", this.y_scale.bandwidth())
            .attr("fill", "#69b3a2")
    }
}
