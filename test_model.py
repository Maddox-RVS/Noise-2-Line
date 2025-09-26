from create_dataset import generateData
import matplotlib.pyplot as plt
from train_model import Model
import numpy as np
import pathlib
import torch
import cv2

MODELS_SAVE_PATH: pathlib.Path = pathlib.Path("models_out")

def calculateSimilarity(outputImageTensor: torch.Tensor, predictedImageTensor: torch.Tensor) -> float:
    outputRedPixels: int = 0
    for i in range(outputImageTensor.shape[0]):
        for j in range(outputImageTensor.shape[1]):
            for k in range(outputImageTensor.shape[2]):
                value = outputImageTensor[i, j, k]
                red: float = value[0].item()
                green: float = value[1].item()
                blue: float = value[2].item()
                if red == 1.0 and green == 0.0 and blue == 0.0:
                    outputRedPixels += 1

    predictedRedPixels: int = 0
    for i in range(predictedImageTensor.shape[0]):
        for j in range(predictedImageTensor.shape[1]):
            for k in range(predictedImageTensor.shape[2]):
                value = predictedImageTensor[i, j, k]
                red: float = value[0].item()
                green: float = value[1].item()
                blue: float = value[2].item()
                if red >= 0.5 and green <= 0.5 and blue <= 0.5:
                    predictedRedPixels += 1

    difference: float = abs(outputRedPixels - predictedRedPixels)
    totalPixels: int = outputImageTensor.shape[0] * outputImageTensor.shape[1]
    similarity: float = 100.0 - ((difference / totalPixels) * 100.0)

    return similarity

def getTestOptions() -> tuple[pathlib.Path, int, bool]:
    modelName: str = ''
    while True:
        modelName = input('Enter the model filename to test: ').strip()
        modelPath: pathlib.Path = MODELS_SAVE_PATH / modelName
        
        if modelPath.exists(): break
        else: print(f'Model file {modelPath} does not exist. Please try again.\n')

    numberOfTestImages: int = 0
    while True:
        try:
            numberOfTestImages = int(input('Enter the number of test images to generate: ').strip())
            
            if numberOfTestImages <= 0:
                print('Please enter a positive integer for the number of test images.\n')
                continue

            break
        except ValueError:
            print('Invalid input. Please enter a valid integer for the number of test images.\n')

    showTestDataFeedback: bool = False
    while True:
        showInput: str = input('Do you want to display test images with predictions? (y/n): ').strip().lower()
        if showInput in ['y', 'yes']:
            showTestDataFeedback = True
            break
        elif showInput in ['n', 'no']:
            showTestDataFeedback = False
            break
        else:
            print('Invalid input. Please enter "y" for yes or "n" for no.\n')

    return modelPath, numberOfTestImages, showTestDataFeedback

def normalizeData(image: np.ndarray) -> np.ndarray:
    normalizedImage: np.ndarray = image.astype(np.float32) / 255.0
    return normalizedImage

def testModel(modelPath: pathlib.Path, testImagesAmount: int, show=False) -> None:
    if not modelPath.exists():
        raise FileNotFoundError(f'Model file not found at {modelPath}.')
    
    print(f'Loading model from {modelPath}...')
    model: Model = Model()
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    print('Generating test dataset...')
    inputImages, outputImages = generateData(testImagesAmount)
    print('Test dataset generation complete.')

    print('Testing model on test dataset...')
    for i, (inputImage, outputImage) in enumerate(zip(inputImages, outputImages)):
        normalizedInputImage: np.ndarray = normalizeData(inputImage)
        normalizedOutputImage: np.ndarray = normalizeData(outputImage)

        inputImageTensor: torch.Tensor = torch.from_numpy(normalizedInputImage).float().unsqueeze(0)
        outputImageTensor: torch.Tensor = torch.from_numpy(normalizedOutputImage).float().unsqueeze(0)

        with torch.no_grad():
            predictedImageTensor: torch.Tensor = model(inputImageTensor)

            similarity: float = calculateSimilarity(outputImageTensor, predictedImageTensor)
            
        if not show:
            print(f'Test Image {i + 1}/{testImagesAmount} - Similarity: {similarity:.2f}%')
        else:
            print(f'Test Image {i + 1}/{testImagesAmount} - Similarity: {similarity:.2f}% - Displaying images...')

            predictedImage: np.ndarray = predictedImageTensor.squeeze(0).numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle('Prediction Comparison', fontsize=16)

            axes[0].imshow(normalizedInputImage, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')

            axes[1].imshow(normalizedOutputImage, cmap='gray')
            axes[1].set_title('Expected Output')
            axes[1].axis('off')

            axes[2].imshow(predictedImage, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            plt.subplots_adjust(wspace=0.3)  # Add space between the images
            plt.tight_layout()
            plt.show()
    print('Model testing complete.')

def main() -> None:
    modelPath, testImagesAmount, showTestDataFeedback = getTestOptions()
    testModel(modelPath, testImagesAmount, showTestDataFeedback)

if __name__ == "__main__":
    main()