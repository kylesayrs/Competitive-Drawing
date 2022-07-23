console.log(document)
const canvas = document.getElementById("paint");
const canvasContext = canvas.getContext("2d");
var mouseHolding = false;

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
