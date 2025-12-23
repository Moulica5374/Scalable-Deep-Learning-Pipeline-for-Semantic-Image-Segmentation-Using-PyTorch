import torch
import numpy as np

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_valid_loss = np.inf

for i in range(epochs):
    train_loss = train_fn(train_loader, model, optimizer)
    val_loss = eval_fn(valid_loader, model, optimizer)

    if val_loss < best_valid_loss:
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved model")
        best_valid_loss = val_loss

    print(
        f"Epoch : {i+1} "
        f"Train_loss :{train_loss} "
        f"Valid_loss :{val_loss}"
    )
