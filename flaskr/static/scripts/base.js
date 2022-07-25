// libraries
pica = pica({ features: ["js"] })

// get canvas elements
const canvas = document.getElementById("draw");
const canvasContext = canvas.getContext("2d");
canvasContext.lineCap = "round";
canvasContext.miterLimit = 1;
const previewCanvas = document.getElementById("preview")
const previewCanvasContext = previewCanvas.getContext("2d")
distanceIndicatorButton = document.getElementById("distanceIndicatorButton")

// global state
allLabels = ["sheep", "dragon", "mona_lisa", "guitar", "pig",
             "tree", "clock", "squirrel", "duck", "jail"]
var targetLabels = allLabels;
var sumPixelsChanged = 0;
var mouseHolding = false;
var inferenceMutex = false;
var lastMouseX = 0;
var lastMouseY = 0;
var totalMouseDistance = 0;
var mouseDistanceLimit = 2000000; // TODO: needs to be proportional to canvas size
                                  // also the count needs to update on resize

canvasContext.lineWidth = 15; // TODO: Proportional to canvas size

// set up prediction chart
confidenceChartMargin = {"top": 20, "right": 30, "bottom": 40, "left": 90}
confidenceChartWidth = 460 - confidenceChartMargin.left - confidenceChartMargin.right
confidenceChartHeight = 400 - confidenceChartMargin.top - confidenceChartMargin.bottom;
var confidenceChartSvg = d3.select("#confidenceChart")
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
// TODO: wrap this in a promise and have game wait until model is initialized
const inferenceSessionPromise = ort.InferenceSession.create(
    modelPath
);

// scale
canvasContext.scale(
    canvas.width / canvas.getBoundingClientRect().width,
    canvas.height / canvas.getBoundingClientRect().height
)

// update mouse distance indicator
// TODO: watch for totalMouseDistance update
function updateDistanceIndicator() {
    document.getElementById("distanceIndicator")
        .innerHTML = "Distance remaining: " + Math.round(mouseDistanceLimit - totalMouseDistance).toString();
}
updateDistanceIndicator()

function getMousePosition(mouseEvent, canvas) {
    const canvasBoundingRect = canvas.getBoundingClientRect();
    mouseX = event.clientX - canvasBoundingRect.left - canvas.offsetLeft + 0.5;
    mouseY = event.clientY - canvasBoundingRect.top - canvas.offsetTop + 0.5;
    return { mouseX, mouseY };
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

function cropCanvasImage(canvasImage) {
    const canvasImageWidth = canvasImage.getWidth();
    const canvasImageHeight = canvasImage.getHeight();
    let bounds = {
        "min_width": canvasImageWidth,
        "max_width": 0,
        "min_height": canvasImageHeight,
        "max_height": 0,
    }
    let foundAnyPixels = false
    for (let y = 0; y < canvasImageHeight; y++) {
        for (let x = 0; x < canvasImageWidth; x++) {
        	let alpha = canvasImage.getAlphaComponent(x, y);
            if (alpha > 0) {
                bounds["min_width"] = Math.min(bounds["min_width"], x);
                bounds["max_width"] = Math.max(bounds["max_width"], x);
                bounds["min_height"] = Math.min(bounds["min_height"], y);
                bounds["max_height"] = Math.max(bounds["max_height"], y);
                foundAnyPixels = true
            }
        }
    }

    if (foundAnyPixels) {
        const cropMidpoint = [(bounds["max_width"] + bounds["min_width"]) / 2,
                              (bounds["max_height"] + bounds["min_height"]) / 2]
        const cropDiameter = Math.floor(Math.max(bounds["max_width"] - bounds["min_width"],
                                                 bounds["max_height"] - bounds["min_height"]))
        const cropRadius = cropDiameter / 2

        const cropX = Math.floor(Math.max(cropMidpoint[0] - cropRadius, 0))
        const cropY = Math.floor(Math.min(cropMidpoint[1] - cropRadius, canvasImageWidth))

        croppedCanvasImage = new MarvinImage(cropDiameter, cropDiameter, MarvinImage.COLOR_MODEL_BINARY)
        Marvin.crop(
            canvasImage,
            croppedCanvasImage,
            cropX,
            cropY,
            cropDiameter,
            cropDiameter,
        );
        return croppedCanvasImage
    } else {
        console.log("WARNING: did not find any pixels when cropping canvas image")
        return canvasImage
    }
}


function imageData2BWData(imageData) {
    // need to get alpha channel because MarvinJ's getColorComponent is broken
    alphaChannelBuffer = []
    for (i = 0; i < imageData.data.length; i += 4) {
        alphaChannelBuffer.push(imageData.data[i + 3]);//resizedImageData[i + 2])
    }

    return new Float32Array(alphaChannelBuffer)
}

async function updatePreview(canvasContext, callbackFn) {
    canvas.toBlob((blob) => {
        canvasBlobUrl = URL.createObjectURL(blob)
        canvasImage = new MarvinImage(canvas.width, canvas.height, MarvinImage.COLOR_MODEL_BINARY)
        canvasImage.load(canvasBlobUrl, async () => {
            // crop canvas image to exactly bounds
            cropedCanvasImage = cropCanvasImage(canvasImage)

            // scale to 26x26 using pica (marvinj's rescaling sucks)
            let image26Data = await pica.resizeBuffer({
                "src": cropedCanvasImage.imageData.data,
                "width": cropedCanvasImage.imageData.width,
                "height": cropedCanvasImage.imageData.height,
                "toWidth": 26,
                "toHeight": 26
            })
            image26Data = new Uint8ClampedArray(image26Data)
            const image26ImageData = new ImageData(image26Data, 26, 26)

            // place onto 28x28 with 1x1 padding
            previewCanvasContext.putImageData(image26ImageData, 1, 1)
            const previewCanvasImageData = previewCanvasContext.getImageData(0, 0, previewCanvas.width, previewCanvas.height)
            callbackFn(previewCanvasImageData)
        })
    })
}

function normalize(arr, minNorm=0, maxNorm=1) {
    const minimumValue = Math.min.apply(Math, arr)
    arr = arr.map((value) => value - minimumValue, minNorm)
    const maxValue = Math.max.apply(Math, arr)
    const ratio = maxValue * maxNorm

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

async function clientInferImage(callbackFn=null) {
    updatePreview(canvasContext, async (previewImageData) => {
        // grab data from preview canvas
        const imageDataBuffer = imageData2BWData(previewImageData)

        // create input
        const model_input = new ort.Tensor(
            imageDataBuffer,
            [1, 1, 28, 28]
        );

        // perform inference
        // TODO: remove promise when game loads after inference session is loaded
        inferenceSession = await inferenceSessionPromise
        const model_outputs = await inferenceSession.run({ "input": model_input })

        // normalize scores
        const model_outputs_normalized = normalize(model_outputs.output.data, 0, 1)

        // filter to target outputs
        filteredLabels = []
        filteredOutputs = []
        for (let i = 0; i < allLabels.length; i++) {
            if (targetLabels.includes(allLabels[i])) {
                filteredLabels.push(allLabels[i])
                filteredOutputs.push(model_outputs_normalized[i])
            }
        }

        // apply softmax
        const model_confidences = softmax(filteredOutputs, factor=7)

        // update chart
        chartData = []
        for (i = 0; i < filteredLabels.length; i++) {
            chartData.push({"label": filteredLabels[i], "value": model_confidences[i]})
        }
        updatePredictionChart(chartData)

        if (callbackFn) {
            callbackFn()
        }
    })
}

async function serverInferImage(callbackFn=null) {
    return; // TODO: implement

    response = await fetch(
        inferenceUrl,
        {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": JSON.stringify({"imageData": canvasImageData})
        }
    )
    if (callbackFn) {
        callbackFn()
    }
    // TODO on response success, update some prediction element
}

canvas.onmousedown = (mouseEvent) => {
    if (mouseDistanceLimit - totalMouseDistance > 1) {
        let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);

        canvasContext.beginPath();
        canvasContext.moveTo(mouseX, mouseY);

        mouseHolding = true;
        lastMouseX = mouseX;
        lastMouseY = mouseY;
    }
}

function onMouseEnd(_mouseEvent) {
    mouseHolding = false;
    sumPixelsChanged = 0;

    // TODO: await on mutex
    if (!inferenceMutex) {
        inferenceMutex = true;
        clientInferImage(() => {
            inferenceMutex = false;
        })
    }

    serverInferImage()
}

canvas.onmouseup = onMouseEnd
canvas.onmouseout = onMouseEnd

canvas.onmousemove = async (mouseEvent) => {
    if (mouseHolding) {
        let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);
        let strokeDistance = Math.hypot(mouseX - lastMouseX, mouseY - lastMouseY)

        // if overreach, interpolate on line to match remaining distance
        if (totalMouseDistance + strokeDistance > mouseDistanceLimit) {
            const distanceRemaining = mouseDistanceLimit - totalMouseDistance
            const theta = Math.asin((mouseY - lastMouseY) / strokeDistance)

            mouseX = Math.cos(theta) * distanceRemaining + lastMouseX
            mouseY = Math.sin(theta) * distanceRemaining + lastMouseY

            // theoretically this should match perfectly
            strokeDistance = Math.hypot(mouseX - lastMouseX, mouseY - lastMouseY)
            //strokeDistance = distanceRemaining

            mouseHolding = false;
        }

        canvasContext.lineTo(mouseX, mouseY);
        canvasContext.stroke();

        totalMouseDistance += strokeDistance
        updateDistanceIndicator()

        lastMouseX = mouseX;
        lastMouseY = mouseY;

        if (!inferenceMutex) {
            inferenceMutex = true;
            await clientInferImage(() => {
                inferenceMutex = false;
            })
        }
    }
};

distanceIndicatorButton.onclick = (_event) => {
    totalMouseDistance = 0;
    updateDistanceIndicator()
}

//TODO: window.onresize = () =>
