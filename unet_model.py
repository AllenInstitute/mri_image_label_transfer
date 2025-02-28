import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, out_channels=64):
        super(SimpleUNet, self).__init__()
        # Encoder: two downsampling steps
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Bottleneck: deeper features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.enc1(x)      # [B, base_channels, H/2, W/2]
        x = self.enc2(x)      # [B, base_channels*2, H/4, W/4]
        features = self.bottleneck(x)  # [B, out_channels, H/4, W/4]
        return features

def extract_features_unet(model, img, alpha = 0.5):
    """
    Given a grayscale image (numpy array of shape [H, W]),
    extract per-pixel features using the U-Net model,
    upsample them to the original resolution, and
    augment with normalized spatial coordinates.
    """
    H, W = img.shape
    # Convert image to tensor and add batch and channel dimensions.
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        feat = model(img_tensor)  # shape: [1, 64, H/4, W/4]
        feat_up = F.interpolate(feat, size=img.shape, mode='bilinear', align_corners=False)
    # Reshape features to (H*W, feature_dim)
    features = feat_up.squeeze(0).permute(1, 2, 0).reshape(H * W, -1).cpu().numpy()
    
    # Create normalized spatial coordinates.
    y_coords, x_coords = np.indices((H, W))
    x_coords = (x_coords.flatten() / W) * alpha  # scale x coordinate
    y_coords = (y_coords.flatten() / H) * alpha  # scale y coordinate
    spatial_coords = np.stack([y_coords, x_coords], axis=1)
    
    # Concatenate spatial coordinates to the features.
    augmented_features = np.concatenate([features, spatial_coords], axis=1)
    return augmented_features