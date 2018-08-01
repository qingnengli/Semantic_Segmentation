set -e

WORK_DIR=/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/deeplab
TRAIN_LOGDIR=/home/amax/SIAT/Semantic_Segmentation/train
EXPORT_LOGDIR=/home/amax/SIAT/Semantic_Segmentation/export_model


CUDA_VISIBLE_DEVICES="-1" TF_CPP_MIN_LOG_LEVEL="2" python "${WORK_DIR}"/export_model.py \
  --checkpoint_path="${TRAIN_LOGDIR}"/model.ckpt-90000 \
  --export_path="${EXPORT_LOGDIR}"/frozen_graph.pb \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
  --decoder_output_stride=4

