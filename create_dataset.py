import numpy as np
import pathlib
import cv2
import shutil

DATASET_SIZE: int = 1000
IMAGE_SIZE: int = 16

def generateData(datasetSize: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputImages: list[np.ndarray] = []
    outputImages: list[np.ndarray] = []

    print('Generating dataset...')
    for i in range(datasetSize):    
        inputImage: np.ndarray = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
        
        redPixelProb: float = 0.90
        redPixels: list[tuple[int, int]] = []

        x, y = np.random.randint(0, IMAGE_SIZE, size=2)
        redPixels.append((x, y))
        while (np.random.rand() < redPixelProb):
            x, y = np.random.randint(0, IMAGE_SIZE, size=2)
            redPixels.append((x, y))

        for (x, y) in redPixels:
            inputImage[x, y] = [0, 0, 0]
        inputImages.append(inputImage)

        outputImage: np.ndarray = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
        for (x, y) in redPixels:
            outputImage[x, y] = [255, 0, 0]
        outputImages.append(outputImage)

        print(f'Generated {i + 1}/{datasetSize} images', end='\r')

    print('\nDataset generation complete.')
    return inputImages, outputImages

def saveImages(images: list[np.ndarray], path: pathlib.Path) -> None:
    print(f'Saving images to {path}...')
    path.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        bgrImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path / f"image_{i:05d}.png"), bgrImage)
        print('Saved {}/{} images.'.format(i + 1, len(images)), end='\r')
    print('\nImage saving complete.')

def removeDatasetDirectory(path: pathlib.Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

def main() -> None:
    datasetPath = pathlib.Path("dataset")
    removeDatasetDirectory(datasetPath)

    inputImagesPath = pathlib.Path(datasetPath / "input_images")
    outputImagesPath = pathlib.Path(datasetPath / "output_images")

    dataset: tuple[list[np.ndarray], list[np.ndarray]] = generateData(DATASET_SIZE)
    inputImages = dataset[0]
    outputImages = dataset[1]

    saveImages(inputImages, inputImagesPath)
    saveImages(outputImages, outputImagesPath)

if __name__ == "__main__":
    main()