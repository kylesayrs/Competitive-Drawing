export function imageToImageData(image) {
    var imageDataBuffer = []
    const add_alpha = image[0][0].length == 3
    for (let y = 0; y < image.length; y++) {
        for (let x = 0; x < image[y].length; x++) {
            for (let c = 0; c < image[y][x].length; c++) {
                imageDataBuffer.push(image[y][x][c])
            }
            if (add_alpha) {
                imageDataBuffer.push(255)
            }
        }
    }

    const imageData = new ImageData(
        new Uint8ClampedArray(imageDataBuffer),
        28,
        28,
    )

    return imageData;
}

export async function resizeImageData(srcImageData, dstImageSize) {
    const dstImageData = await pica.resizeBuffer({
        "src": srcImageData.data,
        "width": srcImageData.width,
        "height": srcImageData.height,
        "toWidth": dstImageSize[0],
        "toHeight": dstImageSize[1]
    })
    const dstImageDataArray = new Uint8ClampedArray(dstImageData)
    return new ImageData(dstImageDataArray, dstImageSize[0], dstImageSize[1])
}
