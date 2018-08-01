## Semantic_Segmentation
DeepLab for VOC2017 datasets in Tensorflow

# Pre-trained model
Deeplab pre-trained model (.pb and .ckpt) are stored in respective folder

# Dataset
The semantic segmentation is based on PASCAL VOC2007 datasets (21 classes)
This Dataset can be downloaded from the website: http://host.robots.ox.ac.uk/pascal/VOC/
Here, SemantcClasses images are used as labels (necessary to convert to 'L' channel for validation)

# Tensorflow Deeplab API
The tensorflow API--tensorflow/Models/research/deeplab is the main tool for semantic segmentation. The folder--'master' saves the corresponding shell file (.sh) to train/val/vis/export model. Their results are outputed on the respective folder--train/val/vis.
