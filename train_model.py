from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import pathlib
import torch
import math
import cv2

MODELS_SAVE_PATH: pathlib.Path = pathlib.Path("models_out")
EPOCHS: int = 350
LEARNING_RATE: float = 0.00005

def loadImages(imagesPath: pathlib.Path) -> list[np.ndarray]:
    images: list[np.ndarray] = []

    if not imagesPath.exists():
        raise FileNotFoundError(f'The path {imagesPath} does not exist.')
    
    imageFiles = list(imagesPath.glob("**/*.png"))
    if not imageFiles:
        raise FileNotFoundError(f'No image files found in {imagesPath}.')

    print(f'Loading images from {imagesPath}...')
    for i, imageFile in enumerate(imageFiles):
        image: np.ndarray = cv2.imread(str(imageFile))
        if image is None:
            raise ValueError(f'Failed to read image file {imageFile} at index {i}.')
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f'Loaded {i + 1}/{len(imageFiles)} images.', end='\r')
        
        images.append(image)

    print('\nImage loading complete.')
    return images

def openDataset(inputImagesPath: pathlib.Path, outputImagesPath: pathlib.Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputImages: list[np.ndarray] = loadImages(inputImagesPath)
    print('\n')

    outputImages: list[np.ndarray] = loadImages(outputImagesPath)
    print( '\n')

    return inputImages, outputImages

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, inputImages: pathlib.Path, outputImages: pathlib.Path):
        self.inputImages, self.outputImages = openDataset(inputImages, outputImages)

    def __len__(self) -> int:
        return len(self.inputImages)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputImage: np.ndarray = self.inputImages[idx]
        outputImage: np.ndarray = self.outputImages[idx]

        inputTensor: torch.Tensor = torch.from_numpy(inputImage).float()
        outputTensor: torch.Tensor = torch.from_numpy(outputImage).float()

        inputTensor /= 255.0
        outputTensor /= 255.0

        return inputTensor, outputTensor
    
def createDataLoader(inputImagesPath: pathlib, outputImagesPath: pathlib, batchSize: int) -> DataLoader:
    dataset: ImageDataset = ImageDataset(inputImagesPath, outputImagesPath)
    dataloader: DataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    return dataloader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten: nn.Flatten = nn.Flatten()
        self.convLayers: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.convLayers(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
def trainModel(model: nn.Module, dataloader: DataLoader, epochs: int, learningRate: float) -> None:
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learningRate)
    criterion: nn.MSELoss = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        runningLoss: float = 0.0

        print(f'<----- Starting epoch {epoch + 1}/{epochs} ----->')

        for batchIndex, (inputs, targets)  in enumerate(dataloader):
            optimizer.zero_grad()
            modelOutputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(modelOutputs, targets)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batchIndex + 1}/{len(dataloader)}]', end='\r' if batchIndex + 1 < len(dataloader) else '\n')

        avgLossEpoch = runningLoss / len(dataloader)        
        print(f'---> Epoch [{epoch + 1}/{epochs}], Average Loss: {avgLossEpoch:.4f}\n')

    print(f'<----- Training complete. ----->\n')

def saveModel(model: nn.Module, saveDir: pathlib.Path, modelName: str) -> None:
    saveDir.mkdir(parents=True, exist_ok=True)
    savePath = saveDir / f'{modelName}.pth'

    torch.save(model.state_dict(), savePath)
    print(f"Model {modelName} saved to {saveDir}.")

def main() -> None:
    datasetPath: pathlib.Path = pathlib.Path("dataset")
    inputImagesPath: pathlib.Path = datasetPath / "input_images"
    outputImagesPath: pathlib.Path = datasetPath / "output_images"

    dataloader: DataLoader = createDataLoader(inputImagesPath, outputImagesPath, batchSize=32)
    model: Model = Model()
    trainModel(model, dataloader, epochs=EPOCHS, learningRate=LEARNING_RATE)

    modelName: str = ''
    while True:
        modelName = input("Enter a name for the saved model: ").strip()
        if (MODELS_SAVE_PATH / f'{modelName}.pth').exists():
            print(f"A model with the name '{modelName}' already exists. Please choose a different name.\n")
            continue
        break

    saveModel(model, MODELS_SAVE_PATH, modelName)

if __name__ == "__main__":
    main()