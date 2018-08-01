set -e
WORK_DIR=/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/deeplab
TRAIN_LOGDIR=/home/amax/SIAT/Semantic_Segmentation/train
EVAL_LOGDIR=/home/amax/SIAT/Semantic_Segmentation/val
TFRECORDS_DIR=/home/amax/SIAT/Semantic_Segmentation/tfrecord

CUDA_VISIBLE_DEVICES="0" TF_CPP_MIN_LOG_LEVEL="2" python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_batch_size=4 \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --eval_interval_secs=600 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset="pascal_voc_seg" \
  --dataset_dir="${TFRECORDS_DIR}" 


