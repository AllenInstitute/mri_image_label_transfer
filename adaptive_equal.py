import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import nibabel as nib


img_nissl = nib.load('/allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Inkar/2025_Hackathon/data/NISSL_IMAGE.nii')
data_nissl = img_nissl.get_fdata()
slice_199_nissl = data_nissl[:, :, 199]
slice_199_nissl = slice_199_nissl.astype(np.uint8)
clahef = clahe.apply(slice_199_nissl)

image_src = cv2.imread("./2025_Hackathon/data/slice_199_nissl.png")
image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahef = clahe.apply(image_src)
plt.imshow(clahef, cmap='gray')
plt.axis('off')
plt.savefig('./2025_Hackathon/data/slice_199_nissl_adap_equalized_8x8.png', bbox_inches='tight', pad_inches=0)
plt.close()
np.save("./2025_Hackathon/data/slice_199_nissl_adap_equalized_8x8.npy",clahef)

image_src = cv2.imread("./2025_Hackathon/data/slice_200_nissl.png")
image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahef = clahe.apply(image_src)
plt.imshow(clahef, cmap='gray')
plt.axis('off')
plt.savefig('./2025_Hackathon/data/slice_200_nissl_adap_equalized_8x8.png', bbox_inches='tight', pad_inches=0)
plt.close()
np.save("./2025_Hackathon/data/slice_200_nissl_adap_equalized_8x8.npy",clahef)