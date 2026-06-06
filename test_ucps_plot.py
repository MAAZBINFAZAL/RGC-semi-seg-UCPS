import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def uncertainty_weight(logits):
    prob = F.softmax(logits.detach(), dim=1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
    entropy = entropy / np.log(prob.shape[1])
    return 1.0 - entropy


loss_fn = nn.CrossEntropyLoss(reduction="none")

epochs = 20
original_losses = []
ucps_losses = []
confidence_scores = []

for epoch in range(epochs):
    batch = 2
    classes = 3
    depth = 16
    height = 64
    width = 64

    # Simulated 3D AO-OCT prediction tensors
    pred_1 = torch.randn(batch, classes, depth, height, width, requires_grad=True)
    pred_2 = torch.randn(batch, classes, depth, height, width, requires_grad=True)

    _, pseudo_1 = torch.max(pred_1.detach(), dim=1)
    _, pseudo_2 = torch.max(pred_2.detach(), dim=1)

    # Original CPS
    original_loss_1 = loss_fn(pred_1, pseudo_2).mean()
    original_loss_2 = loss_fn(pred_2, pseudo_1).mean()
    original_loss = original_loss_1 + original_loss_2

    # UCPS
    conf_1 = uncertainty_weight(pred_1)
    conf_2 = uncertainty_weight(pred_2)

    loss_map_1 = loss_fn(pred_1, pseudo_2)
    loss_map_2 = loss_fn(pred_2, pseudo_1)

    ucps_loss_1 = (loss_map_1 * conf_2).mean()
    ucps_loss_2 = (loss_map_2 * conf_1).mean()
    ucps_loss = ucps_loss_1 + ucps_loss_2

    ucps_loss.backward()

    original_losses.append(original_loss.item())
    ucps_losses.append(ucps_loss.item())
    confidence_scores.append(((conf_1.mean() + conf_2.mean()) / 2).item())

print("UCPS synthetic 3D AO-OCT test completed.")
print("Final original CPS loss:", original_losses[-1])
print("Final UCPS loss:", ucps_losses[-1])
print("Mean confidence:", np.mean(confidence_scores))

plt.figure()
plt.plot(range(1, epochs + 1), original_losses, label="Original CPS Loss")
plt.plot(range(1, epochs + 1), ucps_losses, label="Uncertainty-Aware CPS Loss")
plt.xlabel("Synthetic Test Iteration")
plt.ylabel("Loss")
plt.title("Original CPS vs Uncertainty-Aware CPS")
plt.legend()
plt.grid(True)
plt.savefig("ucps_loss_comparison.png", dpi=300)

plt.figure()
plt.plot(range(1, epochs + 1), confidence_scores, label="Mean UCPS Confidence")
plt.xlabel("Synthetic Test Iteration")
plt.ylabel("Mean Confidence")
plt.title("Entropy-Based Confidence During UCPS Test")
plt.legend()
plt.grid(True)
plt.savefig("ucps_confidence_curve.png", dpi=300)

print("Saved graphs:")
print("ucps_loss_comparison.png")
print("ucps_confidence_curve.png")