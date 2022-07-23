// initialize canvas and drawing
const canvas = document.getElementById("draw");
const canvasContext = canvas.getContext("2d");
canvasContext.lineWidth = 10;
canvasContext.lineCap = "round";
//canvasContext.filter = "blur(0.5px)"
var mouseHolding = false;

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

function inferImage() {
    canvasImageData = canvasContext.getImageData(0, 0, canvas.width, canvas.height)
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
    // TODO on response success, update some prediction element
}

canvas.onmousedown = (mouseEvent) => {
    mouseHolding = true;

    let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);
    canvasContext.beginPath();
    canvasContext.moveTo(mouseX, mouseY);
}

canvas.onmouseup = (_mouseEvent) => {
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
