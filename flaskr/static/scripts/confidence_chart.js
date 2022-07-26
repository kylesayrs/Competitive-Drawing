
export class ConfidenceChart {
    constructor(labels) {
        this.labels = labels

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
            .domain(this.labels)
            .padding(.1);

        this.confidenceChartSvg.append("g")
            .call(d3.axisLeft(this.y_scale))
    }

    updateData(chartData) {
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
