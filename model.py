import torch
import segmentation_models_pytorch as smp


def build_model(encoder="resnet34", weights="imagenet"):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    return model


def load_model(checkpoint_path=None, device="cpu"):
    model = build_model()
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠️  No checkpoint — using raw ImageNet pretrained encoder.")
    model.to(device)
    model.eval()
    return model


def predict_with_tta(model, image_tensor, device="cpu", threshold=0.75):
    import numpy as np
    import torch

    image_tensor = image_tensor.to(device)
    preds = []

    with torch.no_grad():
        for k in range(4): 
            rotated = torch.rot90(image_tensor, k, dims=[2, 3])
            pred = model(rotated)
            unrotated = torch.rot90(pred, -k, dims=[2, 3])
            preds.append(unrotated.squeeze().cpu().numpy())

    avg_pred = np.mean(preds, axis=0)
    mask = (avg_pred > threshold).astype(np.uint8) * 255
    return mask, avg_pred