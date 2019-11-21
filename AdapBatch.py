"""That is an example implementation of how to dynamically adapt the gpu batch size.
Implemented by Victor Zuanazzi"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Example of how to use it with Pytorch
if __name__ == "__main__":

    # #############################################################
    # 1) Initialize the dataset, model, optimizer and loss as usual.
    # Initialize a fake dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.FakeData(size=1_000_000,
                                             image_size=(3, 224, 224),
                                             num_classes=1000,
                                             transform=transform)
    # Note that the DataLoader is not initialized yet.

    # get correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # initialize the model, loss and SGD-based optimizer
    resnet = torchvision.models.resnet152(pretrained=True,
                                          progress=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.01)

    continue_training = True  # criteria to stop the training

    # #############################################################
    # 2) Set parameters for the adaptive batch size
    adapt = True  # while this is true, the algorithm will perform batch adaptation
    gpu_batch_size = 2  # initial gpu batch_size, it can be super small
    train_batch_size = 2048  # the train batch size of desire

    # Modified training loop to allow for adaptive batch size
    while continue_training:

        # #############################################################
        # 3) Initialize dataloader and batch spoofing parameter
        # Dataloader has to be reinicialized for each new batch size.
        print(f"current batch: {gpu_batch_size}")
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=int(gpu_batch_size),
                                                  shuffle=True)

        # Number of repetitions for batch spoofing
        repeat = max(1, int(train_batch_size / gpu_batch_size))

        try:  # This will make sure that training is not halted when the batch size is too large

            # #############################################################
            # 4) Epoch loop with batch spoofing

            optimizer.zero_grad()  # done before training because of batch spoofing.

            for i, (x, y) in tqdm(enumerate(trainloader)):

                y_pred = resnet(x.to(device))
                loss = criterion(y_pred, y.to(device))
                loss.backward()

                # batch spoofing
                if not i % repeat:
                    optimizer.step()
                    optimizer.zero_grad()

                # #############################################################
                # 5) Adapt batch size while no RuntimeError is rased.
                # Increase batch size and get out of the loop
                if adapt:
                    gpu_batch_size *= 2
                    break

                # Stopping criteria for training
                if i > 100:
                    continue_training = False

        # #############################################################
        # 6) After the largest batch size is found, the training progresses with the fixed batch size.
        # CUDA out of memory is a RuntimeError, the moment we will get to it when our batch size is too large.
        except RuntimeError as run_error:
            # This implementation only allows for powers of 2. Which can be seen as rough tuning. Extension to fine
            # tunning will require a few more lines of code, but should also be possible.
            gpu_batch_size /= 2  # resize the batch size for the biggest that works in memory
            adapt = False  # turn off the batch adaptation

            # Number of repetitions for batch spoofing
            repeat = max(1, int(train_batch_size / gpu_batch_size))

            print(f"largest batch size found = {gpu_batch_size}, spoofing repetitions = {repeat}")

            # Manual check if the RuntimeError was caused by the CUDA or something else.
            print(f"---\nRuntimeError: \n{run_error}\n---\n Is it a cuda error?")
