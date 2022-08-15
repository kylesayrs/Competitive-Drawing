export function gradCamImageToImageData(gradCamImage) {
    var imageDataBuffer = []
    for (let y = 0; y < gradCamImage.length; y++) {
        for (let x = 0; x < gradCamImage[y].length; x++) {
            for (let c = 0; c < gradCamImage[y][x].length; c++) {
                imageDataBuffer.push(gradCamImage[y][x][c])
            }
            imageDataBuffer.push(255)
        }
    }

    const imageData = new ImageData(
        new Uint8ClampedArray(imageDataBuffer),
        28,
        28,
    )

    return imageData;
}
