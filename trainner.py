def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    # print("size:", size)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # print(f"X :{X.shape}")
        # print(f"batch:{batch}")
        # print("y length:",y.shape)
        # Compute prediction error
        pred = model(X)
        # print(f"pred:{pred.shape}")
        # print(f"y shape:{y.shape}")
        loss = loss_fn(pred,  y )
        # #
        # # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")