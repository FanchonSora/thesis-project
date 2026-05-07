import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

seg_path = "/home/nvsinh1/brats_segmentation/thesis-project/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz"

nii = nib.load(seg_path)
seg = nii.get_fdata()

print("Shape:", seg.shape)
print("Unique labels:", np.unique(seg))

# slice có tumor lớn nhất
areas = [np.sum(seg[:, :, i] > 0) for i in range(seg.shape[2])]
best_slice = np.argmax(areas)

print("Best slice:", best_slice)

mask = seg[:, :, best_slice]

colors = [
    (0,0,0,0),
    (1.0,0.6,0.6,0.8),   # label 1
    (0.4,1.0,0.8,0.7),   # label 2
    (0.8,0.7,1.0,0.8)    # label 3
]

cmap = ListedColormap(colors)

plt.figure(figsize=(7,7))
plt.imshow(mask.T, origin="lower", cmap=cmap)
plt.axis("off")
plt.tight_layout()

save_path = "tumor_slice.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
print("Saved:", save_path)