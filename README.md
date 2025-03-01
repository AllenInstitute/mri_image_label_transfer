# mri_image_label_transfer
Visit the application website (deployed): https://alleninstitute-mri-image-label-transfer-my-app-6mlgh3.streamlit.app/
Watch the Demo Video: https://drive.google.com/file/d/1ZLDseIsR0dr1hZUTEyb7215lge7TRJAk/view?usp=share_link 

Project Overview: This project was designed as a pilot for transferring brain region segmentation labels from sparsely labeled image volumes. Brain annotation is a time-intensive task that requires manual pixel-by-pixel labeling of structures defined by the human brain ontology. Because of time constraints, the most up-to-date brain annotation only covers ~10% of the brain volume. This project’s goal was to develop and deploy a model that transferred labeled sections of the Nissl human brain atlas to unlabeled neighbors. ​
We implemented a semi-supervised algorithm based on a kernel-density function described in Delalleau et al. (2005). We first trained a CNN on the voxel values of a Nissl-stained section and a paired segmentation volume. We then used the tensor embeddings to estimate the label in an unlabeled neighboring section. Our implementation was successful in labeling high contrast features such as the cortical grey matter. However, hyperparameter tuning is necessary for lower contrast regions before scaling to the whole brain volume. ​
​
This project represents an important first step in label transfer for anatomical atlas images and will be useful for future atlas segmentation efforts in AIBS. ​
