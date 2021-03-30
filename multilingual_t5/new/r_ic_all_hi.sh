export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=r_ic_all_hi
export TPU_SIZE=v3-8
export TASK=r_ic_all_hi
export DATA_DIR="${BUCKET}/ai4b-models/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/ai4b-models/mT5/${TASK}/model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"