// libraries
pica = pica({ features: ["js"] })

// initialize canvas and drawing
const canvas = document.getElementById("draw");
const canvasContext = canvas.getContext("2d");
canvasContext.lineWidth = 20;
canvasContext.lineCap = "round";
canvasContext.miterLimit = 1;
const previewCanvas = document.getElementById("preview")
const previewCanvasContext = previewCanvas.getContext("2d")
//previewCanvasContext.filter = "blur(0.5px)"
var mouseHolding = false;

// set up prediction chart
allLabels = ['sheep', 'dragon', 'mona_lisa', 'guitar', 'pig', 'tree', 'clock', 'squirrel', 'duck', 'jail']
confidenceChartMargin = {"top": 20, "right": 30, "bottom": 40, "left": 90}
confidenceChartWidth = 460 - confidenceChartMargin.left - confidenceChartMargin.right
confidenceChartHeight = 400 - confidenceChartMargin.top - confidenceChartMargin.bottom;
var confidenceChartSvg = d3.select("#confidence-chart")
    .append("svg")
        .attr("width", confidenceChartWidth + confidenceChartMargin.left + confidenceChartMargin.right)
        .attr("height", confidenceChartHeight + confidenceChartMargin.top + confidenceChartMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + confidenceChartMargin.left + "," + confidenceChartMargin.top + ")");

var x_scale = d3.scaleLinear()
    .domain([0, 1])
    .range([0, confidenceChartWidth]);
confidenceChartSvg.append("g")
    .attr("transform", "translate(0," + confidenceChartHeight + ")")
    .call(d3.axisBottom(x_scale))
    .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

var y_scale = d3.scaleBand()
    .range([0, confidenceChartHeight])
    .domain(allLabels)
    .padding(.1);
confidenceChartSvg.append("g")
    .call(d3.axisLeft(y_scale))

// initialize model inference session
const inferenceSessionPromise = ort.InferenceSession.create(
    modelPath
);

// scale
canvasContext.scale(
    canvas.width / canvas.getBoundingClientRect().width,
    canvas.height / canvas.getBoundingClientRect().height
)

function getMousePosition(mouseEvent, canvas) {
    mouseX = event.clientX - canvas.offsetLeft + 0.5;
    mouseY = event.clientY - canvas.offsetTop + 0.5;
    return { mouseX, mouseY };
}

function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}

function updatePredictionChart(chartData) {
    const dataRects = document.querySelectorAll(".dataRect");
    if (dataRects.length > 0) {
        for (const dataRect of dataRects) {
            dataRect.parentNode.removeChild(dataRect)
        }
    }

    confidenceChartSvg.selectAll()
        .data(chartData)
        .enter()
        .append("rect")
        .attr("class", "dataRect")
        .attr("x", x_scale(0) )
        .attr("y", function(d) { return y_scale(d.label); })
        .attr("width", function(d) { return x_scale(d.value); })
        .attr("height", y_scale.bandwidth())
        .attr("fill", "#69b3a2")
}

function cropImageData(imageData) {
    /*
    min_width = 0;
    max_width = previewCanvas.width;
    min_height = 0;
    max_height = previewCanvas.width;
    width = 0;
    height = 0;
    for (i = 0; i < canvasContext.length; i += 4) {

        //redChannelBuffer.push(canvasContext[i + 3]);//resizedImageData[i + 2])
    }
    */

    return imageData
}

async function updatePreview(canvasContext) {
    //canvasImage = getCroppedImageData(canvasContext)
    //console.log(canvasImage)
    canvasImageData = canvasContext.getImageData(0, 0, canvas.width, canvas.height)
    croppedImageData = cropImageData(canvasImageData)
    canvasImageBitmap = await createImageBitmap(croppedImageData)
    console.log(canvasImageBitmap)

    await pica.resize(canvasImageBitmap, previewCanvas)
    previewImageData = previewCanvasContext.getImageData(0, 0, previewCanvas.width, previewCanvas.height).data

    redChannelBuffer = []
    for (i = 0; i < previewImageData.length; i += 4) {
        redChannelBuffer.push(previewImageData[i + 3]);//resizedImageData[i + 2])
    }

    return new Float32Array(redChannelBuffer)
}

function normalize(arr) {
    const minimum = Math.min.apply(Math, arr)
    arr = arr.map((v) => v - minimum)
    const ratio = Math.max.apply(Math, arr)

    for ( i = 0; i < arr.length; i++ ) {
        arr[i] /= ratio;
    }
    return arr
}

function softmax(arr, factor = 1) {
    const exponents = arr.map((value) => Math.exp(value * factor))
    const total = exponents.reduce((a, b) => a + b, 0);
    return exponents.map((exp) => exp / total);
}

async function inferImage() {
    if (inferenceMode == "client") {
        processedImageData = await updatePreview(canvasContext)
        const model_input = new ort.Tensor(
            processedImageData,
            [1, 1, 28, 28]
        );

        inferenceSession = await inferenceSessionPromise
        const model_outputs = await inferenceSession.run({ "input": model_input })
        const model_outputs_normalized = normalize(model_outputs.output.data)

        // filter to relevant outputs
        targetLabels = ["clock", "sheep"]
        filteredLabels = []
        filteredOutputs = []
        for (let i = 0; i < allLabels.length; i++) {
            if (targetLabels.includes(allLabels[i])) {
                filteredLabels.push(allLabels[i])
                filteredOutputs.push(model_outputs_normalized[i])
            }
        }

        const model_confidences = softmax(filteredOutputs, factor=7)
        chartData = []
        for (i = 0; i < filteredLabels.length; i++) {
            chartData.push({"label": filteredLabels[i], "value": model_confidences[i]})
        }
        updatePredictionChart(chartData)
    }

    if (inferenceMode == "server") {
        response = fetch(
            inferenceUrl,
            {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": JSON.stringify({"imageData": canvasImageData})
            }
        )
    }
    // TODO on response success, update some prediction element
}

canvas.onmousedown = (mouseEvent) => {
    mouseHolding = true;

    let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);
    canvasContext.beginPath();
    canvasContext.moveTo(mouseX, mouseY);
}

canvas.onmouseup = async (_mouseEvent) => {
    if (mouseHolding) {
        inferImage()
    }
    mouseHolding = false;
}

canvas.onmouseout = (_mouseEvent) => {
    if (mouseHolding) {
        inferImage()
    }
    mouseHolding = false;
}

canvas.onmousemove = (mouseEvent) => {
    if (mouseHolding) {
        let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);
        canvasContext.lineTo(mouseX, mouseY);
        canvasContext.stroke();
    }
};

//TODO: window.onresize = () =>
