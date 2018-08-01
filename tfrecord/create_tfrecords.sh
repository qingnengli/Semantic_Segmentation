set -e

RUN_DIR=/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/deeplab/datasets
OUTPUT_DIR=/home/amax/SIAT/Semantic_Segmentation/tfrecord

cd ${RUN_DIR}

# Root path for PASCAL VOC 2012 dataset.
PASCAL_ROOT="/home/amax/SIAT/Data-PASCALVOC"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${PASCAL_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${PASCAL_ROOT}/SegmentationClassRaw"
IMAGE_DIR="${PASCAL_ROOT}/JPEGImages"
LIST_DIR="${PASCAL_ROOT}/ImageSets/Segmentation"

echo "Removing the color map in ground truth annotations..."
echo "Convert the 3-channel png to single channel format"
python remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"
echo "Generate the 4 Train/TrainVal/Val TFRecords"
python build_voc2012_data.py \
  --image_folder=${IMAGE_DIR} \
  --semantic_segmentation_folder=${SEMANTIC_SEG_FOLDER} \
  --list_folder=${LIST_DIR} \
  --image_format='jpg' \
  --output_dir=${OUTPUT_DIR}
