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
gpu_batch_size = 2  # initial gpu batch_size, it can be super small
train_batch_size = 2048  # the train batch size of desire
continue_training = True  # criteria to stop the training

# Example of training loop
while continue_training:

    # Dataloader has to be reinicialized for each new batch size.
    print(f"current batch: {gpu_batch_size}")
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=int(gpu_batch_size),
                                              shuffle=True)

    # Number of repetitions for batch spoofing
    repeat = max(1, int(train_batch_size/gpu_batch_size))

    try:

        optimizer.zero_grad()

        for i, (x, y) in tqdm(enumerate(trainloader)):

            y_pred = resnet(x.to(device))

            loss = criterion(y_pred, y.to(device))
            loss.backward()

            # batch spoofing
            if not i % repeat:
                optimizer.step()
                optimizer.zero_grad()

            # Increase batch size and get out of the loop
            if adapt:
                gpu_batch_size *= 2
                break

            if i > 1000:
                continue_training = False

    # CUDA out of memory is a RuntimeError, the moment we will get to it when our batch size is too large.
    except RuntimeError:
        # This implementation only allows for powers of 2. Which can be seen as rough tuning. Extention to fine tunning
        # will require a few more lines of code, but should also be possible.
        gpu_batch_size /= 2  # resize the batch size for the biggest that works in memory
        print(f"largest batch size found = {gpu_batch_size}")
        adapt = False  # turn off the batch adaptation

