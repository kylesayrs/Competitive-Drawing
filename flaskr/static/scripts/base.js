// libraries
pica = pica({ features: ["js"] })

// initialize canvas and drawing
const canvas = document.getElementById("draw");
const canvasContext = canvas.getContext("2d");
canvasContext.lineWidth = 20;
canvasContext.lineCap = "round";
const previewCanvas = document.getElementById("preview")
const previewCanvasContext = previewCanvas.getContext("2d")
//previewCanvasContext.filter = "blur(0.5px)"
var mouseHolding = false;

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

async function processCanvasImage(canvasImage) {
    await pica.resize(canvas, previewCanvas)
    resizedImageData = previewCanvasContext.getImageData(0, 0, previewCanvas.width, previewCanvas.height).data

    redChannelBuffer = []
    for (i = 0; i < resizedImageData.length; i += 4) {
        redChannelBuffer.push(resizedImageData[i + 3]);//resizedImageData[i + 2])
    }

    return new Float32Array(redChannelBuffer)
}

function softmax(arr) {
    return arr.map(function(value, index) {
      return Math.exp(value) / arr.map( function(y){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}

async function inferImage() {
    canvasImage = canvasContext.getImageData(0, 0, 28, 28)//canvas.width, canvas.height)

    if (inferenceMode == "client") {
        processedImageData = await processCanvasImage(canvasImage)
        const model_input = new ort.Tensor(
            processedImageData,
            [1, 1, 28, 28]
        );

        inferenceSession = await inferenceSessionPromise
        const model_outputs = await inferenceSession.run({ "input": model_input })
        console.log(model_outputs.output.data)
        const model_confidences = softmax(model_outputs.output.data)
        //['sheep', 'dragon', 'mona_lisa', 'guitar', 'pig', 'tree', 'clock', 'squirrel', 'duck', 'jail']
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
        //inferImage()
    }
};

//TODO: window.onresize = () =>
