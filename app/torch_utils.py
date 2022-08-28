import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import io


# load model
class FFNeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenSize, outputSize)
        
    def forward(self, x):
        outputs = self.linear1(x)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        
        return outputs
    
SAVE_MODEL = 'model/mnist_fnn.pth'
input_size = 28*28
# hidden_size = sqrt(input_size * output_size)
hidden_size = 90
output_size = 10
model = FFNeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(SAVE_MODEL))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    MEAN = (0.1307,)
    STD = (0.3081,)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    image = image_tensor.reshape(-1, 28*28)    
    # forward
    output = model(image)
    _, output = torch.max(output, 1)
    return output