// libraries
pica = pica({ features: ["js"] })
import { ConfidenceChart } from "/static/scripts/confidence_chart.js";

// Some sort of "ready to play" message
var socket = io();
socket.on("connect", function() {
    console.log("connected!")
});
socket.emit("join", {"room": window.location.room});

// get canvas elements
const canvas = document.getElementById("draw");
const canvasContext = canvas.getContext("2d");
canvasContext.lineCap = "round";
canvasContext.miterLimit = 1;
const previewCanvas = document.getElementById("preview")
const previewCanvasContext = previewCanvas.getContext("2d")
const distanceIndicatorButton = document.getElementById("distanceIndicatorButton")

// global state
const allLabels = ["sheep", "dragon", "mona_lisa", "guitar", "pig",
             "tree", "clock", "squirrel", "duck", "jail"]
var targetLabels = ["pig", "duck"];
var mouseHolding = false;
var inferenceMutex = false;
var lastMouseX = 0;
var lastMouseY = 0;
var totalMouseDistance = 0;
var mouseDistanceLimit = 150;
canvasContext.lineWidth = 15;

const confidenceChart = new ConfidenceChart(targetLabels)

// initialize model inference session
// TODO: wrap this in a promise and have game wait until model is initialized
const inferenceSessionPromise = ort.InferenceSession.create(
    "/static/models/model.onnx"
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
    var mouseX = event.clientX - canvasBoundingRect.left - canvas.offsetLeft + 0.5;
    var mouseY = event.clientY - canvasBoundingRect.top - canvas.offsetTop + 0.5;
    return { mouseX, mouseY };
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

        const croppedCanvasImage = new MarvinImage(cropDiameter, cropDiameter, MarvinImage.COLOR_MODEL_BINARY)
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
    var alphaChannelBuffer = []
    for (let i = 0; i < imageData.data.length; i += 4) {
        alphaChannelBuffer.push(imageData.data[i + 3]);//resizedImageData[i + 2])
    }

    return new Float32Array(alphaChannelBuffer)
}

async function updatePreview(canvasContext, callbackFn) {
    canvas.toBlob((blob) => {
        const canvasBlobUrl = URL.createObjectURL(blob)
        const canvasImage = new MarvinImage(canvas.width, canvas.height, MarvinImage.COLOR_MODEL_BINARY)
        canvasImage.load(canvasBlobUrl, async () => {
            // crop canvas image to exactly bounds
            const cropedCanvasImage = cropCanvasImage(canvasImage)

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
        const inferenceSession = await inferenceSessionPromise
        const model_outputs = await inferenceSession.run({ "input": model_input })

        // normalize scores
        const model_outputs_normalized = normalize(model_outputs.output.data, 0, 1)

        // filter to target outputs
        var filteredLabels = []
        var filteredOutputs = []
        for (let i = 0; i < allLabels.length; i++) {
            if (targetLabels.includes(allLabels[i])) {
                filteredLabels.push(allLabels[i])
                filteredOutputs.push(model_outputs_normalized[i])
            }
        }

        // apply softmax
        const model_confidences = softmax(filteredOutputs, 7)

        // update chart
        var chartData = []
        for (let i = 0; i < filteredLabels.length; i++) {
            chartData.push({"label": filteredLabels[i], "value": model_confidences[i]})
        }
        confidenceChart.updateData(chartData)

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
}

canvas.onmousedown = (mouseEvent) => {
    if (mouseDistanceLimit - totalMouseDistance > 1) {
        let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);

        // send socket, then move
        canvasContext.beginPath();
        canvasContext.moveTo(mouseX, mouseY);

        mouseHolding = true;
        lastMouseX = mouseX;
        lastMouseY = mouseY;
    }
}

function onMouseEnd(_mouseEvent) {
    mouseHolding = false;

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
