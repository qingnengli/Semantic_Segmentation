set -e

WORK_DIR=/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/deeplab
TRAIN_LOGDIR=/home/amax/SIAT/Semantic_Segmentation/train
VIS_LOGDIR=/home/amax/SIAT/Semantic_Segmentation/vis
TFRECORDS_DIR=/home/amax/SIAT/Semantic_Segmentation/tfrecord

CUDA_VISIBLE_DEVICES="-1" TF_CPP_MIN_LOG_LEVEL="2" python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_batch_size=32 \
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${TFRECORDS_DIR}" \
  --colormap_type="pascal" \
  --also_save_raw_predictions=False \ 
  --eval_interval_secs=600
