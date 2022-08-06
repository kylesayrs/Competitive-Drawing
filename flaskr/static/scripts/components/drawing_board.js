/*
filename: drawing_board.js
author: Kyle Sayers
details: The DrawingBoard controls the drawing canvas as well as the preview canvas.
         Future implementations will allow a virtual preview canvas rather than
         relying on one already appended to the DOM.
*/

export class DrawingBoard {
    constructor(distanceIndicator=null) {
        this._distanceIndicator = distanceIndicator

        this.canvas = document.getElementById("draw");
        this.canvasContext = this.canvas.getContext("2d");
        this.previewCanvas = document.getElementById("preview")
        this.previewCanvasContext = this.previewCanvas.getContext("2d")

        this.canvasContext.lineCap = "round";
        this.canvasContext.miterLimit = 1;
        this.canvasContext.lineWidth = 7;
        this.canvasContext.scale(
            this.canvas.width / this.canvas.getBoundingClientRect().width,
            this.canvas.height / this.canvas.getBoundingClientRect().height
        )

        this.mouseHolding = false
        this.lastMouseX = 0
        this.lastMouseY = 0

        this.canvas.onmousedown = this.onMouseDown.bind(this)
        this.canvas.onmousemove = this.onMouseMove.bind(this)
        this.canvas.onmouseup = this.onMouseEnd.bind(this)
        this.canvas.onmouseout = this.onMouseEnd.bind(this)

        this.afterMouseEnd = null
        this.afterMouseMove = null
    }

    getMousePosition(mouseEvent) {
        const canvasBoundingRect = this.canvas.getBoundingClientRect();
        var mouseX = event.clientX - canvasBoundingRect.left - this.canvas.offsetLeft + 0.5;
        var mouseY = event.clientY - canvasBoundingRect.top - this.canvas.offsetTop + 0.5;
        return { mouseX, mouseY };
    }

    cropCanvasImage(canvasImage) {
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

    async updatePreview(callbackFn=null) {
        return new Promise((resolve) => {
            this.canvas.toBlob(async (blob) => {
                const canvasBlobUrl = URL.createObjectURL(blob)
                const canvasImage = new MarvinImage(this.canvas.width, this.canvas.height, MarvinImage.COLOR_MODEL_BINARY)
                canvasImage.load(canvasBlobUrl, async () => {
                    // crop canvas image to exactly bounds
                    const cropedCanvasImage = this.cropCanvasImage(canvasImage)

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

                    // TODO: technically there's some time loss here as well as
                    // a potential frame or so where the user can see a blank canvas
                    await this.previewCanvasContext.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);

                    // place onto 28x28 with 1x1 padding
                    this.previewCanvasContext.putImageData(image26ImageData, 1, 1)
                    const previewCanvasImageData = this.previewCanvasContext.getImageData(0, 0, this.previewCanvas.width, this.previewCanvas.height)
                    if (callbackFn) {
                        callbackFn(previewCanvasImageData)
                    } else {
                        resolve(previewCanvasImageData)
                    }
                })
            })
        })
    }

    onMouseDown(mouseEvent) {
        if (this._distanceIndicator == null || this._distanceIndicator.mouseDistanceLimit - this._distanceIndicator.totalMouseDistance > 1) {
            let { mouseX, mouseY } = this.getMousePosition(mouseEvent);

            // send socket, then move
            this.canvasContext.beginPath();
            this.canvasContext.moveTo(mouseX, mouseY);

            this.mouseHolding = true;
            this.lastMouseX = mouseX;
            this.lastMouseY = mouseY;
        }
    }

    onMouseEnd(_mouseEvent) {
        this.mouseHolding = false;
        if (this.afterMouseEnd) {
            this.afterMouseEnd()
        }
    }

    async onMouseMove(mouseEvent) {
        if (this.mouseHolding) {
            let { mouseX, mouseY } = this.getMousePosition(mouseEvent);
            let strokeDistance = Math.hypot(mouseX - this.lastMouseX, mouseY - this.lastMouseY)

            // if overreach, interpolate on line to match remaining distance
            if (this._distanceIndicator && this._distanceIndicator.totalMouseDistance + strokeDistance > this._distanceIndicator.mouseDistanceLimit) {
                const distanceRemaining = this._distanceIndicator.mouseDistanceLimit - this._distanceIndicator.totalMouseDistance
                const theta = Math.asin((mouseY - this.lastMouseY) / strokeDistance)

                mouseX = Math.cos(theta) * distanceRemaining + this.lastMouseX
                mouseY = Math.sin(theta) * distanceRemaining + this.lastMouseY

                // theoretically this should match perfectly
                strokeDistance = Math.hypot(mouseX - this.lastMouseX, mouseY - this.lastMouseY)
                //strokeDistance = distanceRemaining

                this.mouseHolding = false;
            }

            this.canvasContext.lineTo(mouseX, mouseY);
            this.canvasContext.stroke();

            if (this._distanceIndicator) {
                this._distanceIndicator.totalMouseDistance += strokeDistance
            }

            this.lastMouseX = mouseX;
            this.lastMouseY = mouseY;

            if (this.afterMouseMove) {
                this.afterMouseMove()
            }
        }
    }
}
