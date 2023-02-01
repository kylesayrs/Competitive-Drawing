/*
filename: drawing_board.js
author: Kyle Sayers
details: The DrawingBoard controls the drawing canvas as well as the preview canvas.
         Future implementations will allow a virtual preview canvas rather than
         relying on one already appended to the DOM.
*/
import { resizeImageData } from "/static/scripts/helpers.js";

export class DrawingBoard {
    constructor(distanceIndicator, gameConfig) {
        this._distanceIndicator = distanceIndicator
        this.canvasSize = gameConfig.canvasSize
        this.canvasLineWidth = gameConfig.canvasLineWidth
        this.imageSize = gameConfig.imageSize
        this.imagePadding = gameConfig.imagePadding
        this.staticCrop = gameConfig.staticCrop

        this.canvas = document.getElementById("draw");
        this.canvasContext = this.canvas.getContext("2d", {
            alpha: false,
            colorSpace: "srgb",
            desynchronized: false,
            willReadFrequently: true
        })
        this.previewCanvas = document.getElementById("preview")
        this.previewCanvasContext = this.previewCanvas.getContext("2d", {
            alpha: false,
            colorSpace: "srgb",
            desynchronized: false,
            willReadFrequently: true
        })

        this.canvas.setAttribute("width", this.canvasSize)
        this.canvas.setAttribute("height", this.canvasSize)

        this.resetCanvases()

        this.canvasContext.lineCap = "round";
        this.canvasContext.miterLimit = 1;
        this.canvasContext.lineWidth = this.canvasLineWidth;

        this.enabled = true
        this.mouseHolding = false
        this.lastMouseX = 0
        this.lastMouseY = 0

        this.canvas.onmousedown = this.onMouseDown.bind(this)
        this.canvas.onmousemove = this.onMouseMove.bind(this)
        this.canvas.onmouseup = this.onMouseEnd.bind(this)
        this.canvas.onmouseout = this.onMouseEnd.bind(this)
        this.canvas.addEventListener("touchstart", this.canvas.onmousedown, false)
        this.canvas.addEventListener("touchmove", this.canvas.onmousemove, false)
        this.canvas.addEventListener("touchend", this.canvas.onmouseup, false);
        this.canvas.addEventListener("touchcancel", this.canvas.onmouseup, false);

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


    getMousePosition(event) {
        const clientX = (window.TouchEvent && event instanceof TouchEvent) ? event.touches[0].clientX : event.clientX
        const clientY = (window.TouchEvent && event instanceof TouchEvent) ? event.touches[0].clientY : event.clientY

        const canvasBoundingRect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / canvasBoundingRect.width
        const scaleY = this.canvas.height / canvasBoundingRect.height
        const canvasX = clientX - canvasBoundingRect.left
        const canvasY = clientY - canvasBoundingRect.top
        const mouseX = canvasX * scaleX;
        const mouseY = canvasY * scaleY;
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
                if (red < 254) { // TODO: investigate weird 254 values
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
                const croppedCanvasImage = this.staticCrop ? canvasImage : this.cropCanvasImage(canvasImage)

                // scale using pica (marvinj's rescaling sucks)
                const imageInnerSize = this.imageSize - this.imagePadding
                const innerImageData = resizeImageData(croppedCanvasImage.imageData, [imageInnerSize, imageInnerSize])

                // TODO: technically there's some time loss here as well as
                // a potential frame or so where the user can see a blank canvas
                this.resetPreviewBorder();

                // place onto preview canvas with padding
                this.previewCanvasContext.putImageData(await innerImageData, this.imagePadding, this.imagePadding)

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


    onMouseDown(event) {
        if (this._distanceIndicator == null || this._distanceIndicator.distanceRemaining > 1) {
            this._mouseDown(event)
        }
    }


    _mouseDown(event) {
        let { mouseX, mouseY } = this.getMousePosition(event);

        // send socket, then move
        this.canvasContext.beginPath();
        this.canvasContext.moveTo(mouseX, mouseY);

        this.mouseHolding = true;
        this.lastMouseX = mouseX;
        this.lastMouseY = mouseY;
    }


    onMouseEnd(_event) {
        this.mouseHolding = false;
        if (this.afterMouseEnd) {
            this.afterMouseEnd()
        }
    }


    onMouseMove(event) {
        if (this.enabled && this.mouseHolding) {
            this._moveMouse(event)
        }
    }


    _moveMouse(event) {
        let { mouseX, mouseY } = this.getMousePosition(event);
        let strokeDistance = Math.hypot(mouseX - this.lastMouseX, mouseY - this.lastMouseY)

        // if overreach, interpolate on line to match remaining distance
        if (this._distanceIndicator && this._distanceIndicator.mouseDistance + strokeDistance > this._distanceIndicator.mouseDistanceLimit) {
            const distanceRemaining = this._distanceIndicator.distanceRemaining
            const theta = Math.asin((mouseY - this.lastMouseY) / strokeDistance)

            mouseX = Math.cos(theta) * distanceRemaining + this.lastMouseX
            mouseY = Math.sin(theta) * distanceRemaining + this.lastMouseY

            strokeDistance = Math.hypot(mouseX - this.lastMouseX, mouseY - this.lastMouseY)
            // if all is well, then these should match with high precision
            // assert(strokeDistance == distanceRemaining)

            this.mouseHolding = false;
        }

        this.canvasContext.lineTo(mouseX, mouseY);
        this.canvasContext.stroke();

        if (this._distanceIndicator) {
            this._distanceIndicator.mouseDistance += strokeDistance
        }

        this.lastMouseX = mouseX;
        this.lastMouseY = mouseY;

        if (this.afterMouseMove) {
            this.afterMouseMove()
        }
    }


    strokeSampleToMouseEvent(strokeSample) {
        const canvasBoundingRect = this.canvas.getBoundingClientRect()

        const scaleX = (this.canvas.width / canvasBoundingRect.width)
        const scaleY = (this.canvas.height / canvasBoundingRect.height)

        const tmp = {
            clientX: (strokeSample[1] * this.canvas.width) / scaleX + canvasBoundingRect.left,
            clientY: (strokeSample[0] * this.canvas.height) / scaleY + canvasBoundingRect.top,
        }

        return tmp
    }


    async replayStroke(strokeSamples, strokeDurationMS) {
        this.onMouseEnd(this.strokeSampleToMouseEvent(strokeSamples[0]))
        this._mouseDown(this.strokeSampleToMouseEvent(strokeSamples[0]))

        const sampleDurationMS = strokeDurationMS / strokeSamples.length
        for (const i in strokeSamples) {
            this._moveMouse(this.strokeSampleToMouseEvent(strokeSamples[i]))
            await new Promise(r => setTimeout(r, sampleDurationMS));
        }

        const lastSample = strokeSamples[strokeSamples.length - 1]
        this.onMouseEnd(this.strokeSampleToMouseEvent(lastSample))
    }
}
