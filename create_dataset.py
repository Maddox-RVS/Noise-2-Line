import numpy as np
import pathlib
import cv2
import shutil

DATASET_SIZE: int = 1000
IMAGE_SIZE: int = 64

def generateData(datasetSize: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputImages: list[np.ndarray] = []
    outputImages: list[np.ndarray] = []

    print('Generating dataset...')
    for i in range(datasetSize):    
        inputImage: np.ndarray = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
        outputImage: np.ndarray = inputImage.copy()
        
        numLines: int = np.random.randint(1, 11)

        for x in range(IMAGE_SIZE):
            for y in range(IMAGE_SIZE):
                randomRedChannel: int = np.random.randint(0, 256)
                randomGreenChannel: int = np.random.randint(0, 256)
                randomBlueChannel: int = np.random.randint(0, 256)
                inputImage[y, x] = [randomRedChannel, randomGreenChannel, randomBlueChannel]

        for j in range(numLines):
            point1x: int = np.random.randint(0, IMAGE_SIZE)
            point1y: int = np.random.randint(0, IMAGE_SIZE)

            point2x: int = np.random.randint(0, IMAGE_SIZE)
            point2y: int = np.random.randint(0, IMAGE_SIZE)

            thickness: int = np.random.randint(1, 6)

            cv2.line(inputImage, (point1x, point1y), (point2x, point2y), (255, 255, 255), thickness)
            cv2.line(outputImage, (point1x, point1y), (point2x, point2y), (0, 0, 0), thickness)

        for x in range(IMAGE_SIZE):
            for y in range(IMAGE_SIZE):
                pixelChangeProbability: float = np.random.rand()
                if pixelChangeProbability > 0.8:
                    randomRedChannel: int = np.random.randint(0, 256)
                    randomGreenChannel: int = np.random.randint(0, 256)
                    randomBlueChannel: int = np.random.randint(0, 256)
                    inputImage[y, x] = [randomRedChannel, randomGreenChannel, randomBlueChannel]

        inputImages.append(inputImage)
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