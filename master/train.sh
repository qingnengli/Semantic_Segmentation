set -e

RUN_DIR=/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/deeplab
CKPT_DIR=/home/amax/SIAT/Semantic_Segmentation/deeplabv3_pascal_trainval/model.ckpt
LOG_DIR=/home/amax/SIAT/Semantic_Segmentation/train
TFRECORDS_DIR=/home/amax/SIAT/Semantic_Segmentation/tfrecord

cd ${RUN_DIR}

CUDA_VISIBLE_DEVICES="1" TF_CPP_MIN_LOG_LEVEL="2" python train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --atrous_rates=6 \
  --atrous_rates=18 \
  --atrous_rates=12 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_batch_size=4 \
  --training_number_of_steps=90000 \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True \
  --save_summaries_images=True \
  --save_summaries_secs=600 \
  --save_interval_secs=600 \
  --tf_initial_checkpoint=${CKPT_DIR} \
  --train_logdir=${LOG_DIR} \
  --dataset="pascal_voc_seg" \
  --dataset_dir=${TFRECORDS_DIR}
