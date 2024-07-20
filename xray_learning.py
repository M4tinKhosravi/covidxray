import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

def CNN_Model():
    device = torch.device("cpu")
    class_names = ['covid19', 'normal']
    model = models.densenet121(weights=None) # Returns Defined Densenet model with weights trained on ImageNet
    num_ftrs = model.classifier.in_features # Get the number of features output from CNN layer
    model.classifier = nn.Linear(num_ftrs, len(class_names)) # Overwrites the Classifier layer with custom defined layer for transfer learning
    model = model.to(device) # Transfer the Model to GPU if available
    return model


def predict(img_path):
    class_names = ['covid19', 'normal']
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nums, std=std_nums)
    ])

    image = Image.open(img_path).convert('RGB')
    # plt.imshow(np.array(image))
    image_tensor = test_transforms(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(torch.device('cpu'))
    inf_model = CNN_Model()
    inf_model.to(torch.device('cpu'))
    inf_model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    inf_model.eval()
    out = inf_model(input)
    _, preds = torch.max(out, 1)
    idx = preds.cpu().numpy()[0]
    pred_class = class_names[idx]
    return pred_class
