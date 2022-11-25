/*
filename: drawing_board.js
author: Kyle Sayers
details: The DrawingBoard controls the drawing canvas as well as the preview canvas.
         Future implementations will allow a virtual preview canvas rather than
         relying on one already appended to the DOM.
*/
import { resizeImageData } from "/static/scripts/helpers.js";

export class DrawingBoard {
    constructor(distanceIndicator=null, canvasSize=500) {
        this._distanceIndicator = distanceIndicator

        this.canvas = document.getElementById("draw");
        this.canvasContext = this.canvas.getContext("2d", {
            willReadFrequently: true
        })
        this.previewCanvas = document.getElementById("preview")
        this.previewCanvasContext = this.previewCanvas.getContext("2d", {
            willReadFrequently: true
        })

        this.canvas.setAttribute("width", canvasSize)
        this.canvas.setAttribute("height",canvasSize)

        this.resetCanvases()

        this.canvasContext.lineCap = "round";
        this.canvasContext.miterLimit = 1;
        this.canvasContext.lineWidth = this.canvas.width / 70;

        this.enabled = true
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


    resetCanvases() {
        this.canvasContext.rect(0, 0, this.canvas.width, this.canvas.height);
        this.canvasContext.fillStyle = "white";
        this.canvasContext.fill();

        this.previewCanvasContext.beginPath();
        this.previewCanvasContext.rect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
        this.previewCanvasContext.fillStyle = "white";
        this.previewCanvasContext.fill();
    }

    getMousePosition(mouseEvent) {
        const canvasBoundingRect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / canvasBoundingRect.width
        const scaleY = this.canvas.height / canvasBoundingRect.height
        const canvasX = event.clientX - canvasBoundingRect.left
        const canvasY = event.clientY - canvasBoundingRect.top
        const mouseX = scaleX * canvasX;
        const mouseY = scaleY * canvasY;
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
            	let red = canvasImage.getIntComponent0(x, y);
                if (red < 255) {
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

    resetPreviewBorder() {
        this.previewCanvasContext.beginPath();
        this.previewCanvasContext.rect(0, 0, this.previewCanvas.width, 1);
        this.previewCanvasContext.rect(0, 0, 1, this.previewCanvas.height);
        this.previewCanvasContext.rect(this.previewCanvas.width - 1, 0, this.previewCanvas.width, this.previewCanvas.height);
        this.previewCanvasContext.rect(0, this.previewCanvas.height - 1, this.previewCanvas.width, this.previewCanvas.height);
        this.previewCanvasContext.fillStyle = "white";
        this.previewCanvasContext.fill();
    }

    async updatePreview() {
        return new Promise(async (resolve) => {
            const canvasBlobUrl = await this.canvas.toDataURL()
            const canvasImage = new MarvinImage(this.canvas.width, this.canvas.height, MarvinImage.COLOR_MODEL_BINARY)
            canvasImage.load(canvasBlobUrl, async () => {
                // crop canvas image to exactly bounds
                const cropedCanvasImage = this.cropCanvasImage(canvasImage)

                // scale to 26x26 using pica (marvinj's rescaling sucks)
                const image26ImageData = resizeImageData(cropedCanvasImage.imageData, [26, 26])

                // TODO: technically there's some time loss here as well as
                // a potential frame or so where the user can see a blank canvas
                this.resetPreviewBorder();

                // place onto 28x28 with 1x1 padding
                this.previewCanvasContext.putImageData(await image26ImageData, 1, 1)

                // get image data (with border)
                const previewImageData = this.getPreviewImageData()
                resolve(previewImageData)
            })
        })
    }

    getPreviewImageData() {
        return this.previewCanvasContext.getImageData(0, 0, this.previewCanvas.width, this.previewCanvas.height)
    }

    getPreviewImageDataUrl() {
        return this.previewCanvas.toDataURL()
    }

    getCanvasImageDataUrl() {
        return this.canvas.toDataURL()
    }

    async putPreviewImageData(previewCanvasImageData, updateCanvas=false) {
        this.previewCanvasContext.putImageData(previewCanvasImageData, 0, 0)
        if (updateCanvas) {
            if (updateCanvas) {
                const canvasImageData = await resizeImageData(previewCanvasImageData, [this.canvas.width, this.canvas.height])

                this.canvasContext.putImageData(canvasImageData, 0, 0)
            }
        }
        this.previewCanvasContext
    }

    putCanvasImageData(canvasImageData) {
        this.canvasContext.putImageData(canvasImageData, 0, 0)
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
        if (this.enabled && this.mouseHolding) {
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
