import torch

idx = 2

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

image, mask = validset[idx]

with torch.no_grad():
    logits_mask = model(image.to(DEVICE).unsqueeze(0))

pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0
