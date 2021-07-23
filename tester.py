from torch import nn
import torch
def test(model, test_loader, loss_fn,device):
    print("-----Testing Start-----")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += loss_fn(output,  target.to(device) ).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.to(device).view_as(pred)).sum().item() # how many predictions in this batch are correct

    test_loss /= len(test_loader.dataset)

    # Log testing info
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("-----Testing Over-----")