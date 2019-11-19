import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Initialize a fake dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.FakeData(size=1_000_000,
                                         image_size=(3, 224, 224),
                                         num_classes=1000,
                                         transform=transform)

# get correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# initialize the model, loss and SGD-based optimizer
resnet = torchvision.models.resnet152(pretrained=True,
                                      progress=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01)

adapt = True  # while this is true, the algorithm will perform batch adaptation
batch_size = 2  # initial batch size.
continue_training = True  # criteria to stop the training

# Example of training loop
while continue_training:

    # Dataloader has to be reinicialized for each new batch size.
    print(f"current batch: {batch_size}")
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=int(batch_size),
                                              shuffle=True)
    try:
        for i, (x, y) in tqdm(enumerate(trainloader)):
            optimizer.zero_grad()

            y_pred = resnet(x.to(device))

            loss = criterion(y_pred, y.to(device))
            loss.backward()
            optimizer.step()

            if adapt:
                batch_size *= 2
                break

            if i > 3:
                continue_training = False

    # CUDA out of memory is a RuntimeError, the moment we will get to it when our batch size is too large.
    except RuntimeError:
        batch_size /= 2  # resize the batch size for the biggest that works in memory
        print(f"largest batch size found = {batch_size}")
        adapt = False  # turn off the batch adaptation

