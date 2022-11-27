pica = pica({ features: ["js"] })

export function imageToImageData(image, width, height) {
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
        width,
        height,
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

export function normalize(arr, minNorm=0, maxNorm=1) {
    const minimumValue = Math.min.apply(Math, arr)
    arr = arr.map((value) => value - minimumValue, minNorm)
    const maxValue = Math.max.apply(Math, arr)
    const ratio = maxValue * maxNorm

    for (let i = 0; i < arr.length; i++ ) {
        arr[i] /= ratio;
    }
    return arr
}

export function softmax(arr, factor=1) {
    const exponents = arr.map((value) => Math.exp(value * factor))
    const total = exponents.reduce((a, b) => a + b, 0);
    return exponents.map((exp) => exp / total);
}

/*
export function binarizeMarvinImage(marvinImage) {
    for (let pixelStart = 0; pixelStart < marvinImage.imageData.data.length; pixelStart ++) {
        let maxValue = Math.max(marvinImage.imageData.data.slice(pixelStart, pixelStart + 4))
        marvinImage.imageData.data[0] = maxValue
        marvinImage.imageData.data[1] = maxValue
        marvinImage.imageData.data[2] = maxValue
        marvinImage.imageData.data[3] = maxValue
    }

    return marvinImage
}
*/
