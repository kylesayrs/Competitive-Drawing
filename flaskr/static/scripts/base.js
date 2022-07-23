// initialize canvas and drawing
const canvas = document.getElementById("paint");
const canvasContext = canvas.getContext("2d");
canvasContext.lineWidth = 15;
canvasContext.imageSmoothingEnabled = true;
canvasContext.imageSmoothingQuality = "high";
var mouseHolding = false;

// scale
canvasContext.scale(
    imageShape[0] / canvasShape[0],
    imageShape[1] / canvasShape[1],
)

function getMousePosition(mouseEvent, canvas) {
    mouseX = event.clientX - canvas.offsetLeft;
    mouseY = event.clientY - canvas.offsetTop;
    return { mouseX, mouseY };
}

canvas.onmousedown = function(mouseEvent) {
    mouseHolding = true;

    let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);
    canvasContext.beginPath();
    canvasContext.moveTo(mouseX, mouseY);
    console.log(canvasContext.getImageData(0, 0, canvas.width, canvas.height))
}

canvas.onmouseup = function(_mouseEvent) {
    mouseHolding = false;
}

canvas.onmouseout = function(_mouseEvent){
    mouseHolding = false;
}

canvas.onmousemove = function(mouseEvent) {
    if (mouseHolding) {
        let { mouseX, mouseY } = getMousePosition(mouseEvent, canvas);
        canvasContext.lineTo(mouseX, mouseY);
        canvasContext.stroke();
    }
};
