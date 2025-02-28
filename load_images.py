import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image as im 
import numpy as np


img_nissl = nib.load('/allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Inkar/2025_Hackathon/data/NISSL_IMAGE.nii')
data_nissl = img_nissl.get_fdata()
slice_199_nissl = data_nissl[:, :, 199]
plt.imshow(slice_199_nissl, cmap='gray')
plt.axis('off')  # Turn off axes for a cleaner image
plt.savefig('./2025_Hackathon/data/slice_199_nissl.png', bbox_inches='tight', pad_inches=0)
plt.close()

slice_200_nissl = data_nissl[:, :, 200]
plt.imshow(slice_200_nissl, cmap='gray')
plt.axis('off')  # Turn off axes for a cleaner image
plt.savefig('./2025_Hackathon/data/slice_200_nissl.png', bbox_inches='tight', pad_inches=0)
plt.close()

img_brodmann = nib.load('/allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Inkar/2025_Hackathon/data/Brodmann_all.nii')
data_brodmann = img_brodmann.get_fdata()
slice_199_brodmann = data_brodmann[:, :, 199]
plt.imshow(slice_199_brodmann, cmap='gray')
plt.axis('off')  # Turn off axes for a cleaner image
plt.savefig('./2025_Hackathon/data/slice_199_brodmann.png', bbox_inches='tight', pad_inches=0)
plt.close()

slice_200_brodmann = data_brodmann[:, :, 200]
plt.imshow(slice_200_brodmann, cmap='gray')
plt.axis('off')  # Turn off axes for a cleaner image
plt.savefig('./2025_Hackathon/data/slice_200_brodmann.png', bbox_inches='tight', pad_inches=0)
plt.close()


for i in range(data_brodmann.shape[2]):
    print(i)
    nissl_image = im.fromarray((data_nissl[:, :, i]* 255).astype(np.uint16)).convert("L") 
    if np.any(data_brodmann[:, :, i]) != 0:
        brodmann_image = im.fromarray((data_brodmann[:, :, i]* 255).astype(np.uint8)).convert("L")
        brodmann_image.save("./2025_Hackathon/training_images/masks/Brodmann_Nissl_"+str(i)+".png")
        nissl_image.save("./2025_Hackathon/training_images/images/Brodmann_Nissl_"+str(i)+".png")
    else:
        nissl_image.save("./2025_Hackathon/testing_images/Brodmann_Nissl_"+str(i)+".png")
    