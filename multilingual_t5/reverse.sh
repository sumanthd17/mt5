# bn
export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-bn
export TPU_SIZE=v3-8
export TASK=baseline_bn
export MIXTURE=r_baseline_bn
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

# gu
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-gu
export TPU_SIZE=v2-8
export TASK=baseline_gu
export MIXTURE=r_baseline_gu
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

# hi
export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-hi
export TPU_SIZE=v3-8
export TASK=baseline_hi
export MIXTURE=r_baseline_hi
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

#kn
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-kn
export TPU_SIZE=v2-8
export TASK=baseline_kn
export MIXTURE=r_baseline_kn
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

#ml
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-ml
export TPU_SIZE=v2-8
export TASK=baseline_ml
export MIXTURE=r_baseline_ml
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

# mr
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-mr
export TPU_SIZE=v2-8
export TASK=baseline_mr
export MIXTURE=r_baseline_mr
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

# pa
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=sd-r-mt5-baseline-pa
export TPU_SIZE=v2-8
export TASK=baseline_pa
export MIXTURE=r_baseline_pa
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 2000000" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.0005" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 8192)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

# ta
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-ta
export TPU_SIZE=v2-8
export TASK=baseline_ta
export MIXTURE=r_baseline_ta
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"

# te
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-te
export TPU_SIZE=v2-8
export TASK=baseline_te
export MIXTURE=r_baseline_te
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --project_name="${TASK}" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.train_steps = 1262144" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"