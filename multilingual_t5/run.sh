# bn
export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-bn
export TPU_SIZE=v3-8
export TASK=baseline_bn
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

# gu
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-gu
export TPU_SIZE=v2-8
export TASK=baseline_gu
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

# hi
export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-hi
export TPU_SIZE=v3-8
export TASK=baseline_hi
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

#kn
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-kn
export TPU_SIZE=v2-8
export TASK=baseline_kn
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

#ml
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-ml
export TPU_SIZE=v2-8
export TASK=baseline_ml
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

# mr
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-mr
export TPU_SIZE=v2-8
export TASK=baseline_mr
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

# pa
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-pa
export TPU_SIZE=v2-8
export TASK=baseline_pa
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

# ta
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-ta
export TPU_SIZE=v2-8
export TASK=baseline_ta
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

# te
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-mt5-te
export TPU_SIZE=v2-8
export TASK=baseline_te
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

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

# IndicTrans
export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://pre-train
export TPU_NAME=eval
export TPU_SIZE=v3-8
export DATA_DIR="${BUCKET}/transliterated/data"
export MODEL_DIR="${BUCKET}/transliterated/og-model-dir"
export TASK=indic_trans

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}/tfds" \
  --project_name="IndicTrans-pretraining" \
  --gin_file="models/t5.1.1.base.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_param="run.train_steps = 524288" \
  --gin_param="run.save_checkpoints_steps = 50000" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.learning_rate_schedule = @learning_rate_schedules.learning_rate_schedule_noam" \
  --gin_param="learning_rate_schedule_noam.linear_decay_fraction = 0.0" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.mesh_train_dataset_fn" \
  --gin_param="mesh_train_dataset_fn.mixture_or_task_name = %MIXTURE_NAME" \
  --gin_param="mesh_transformer.get_vocabulary.mixture_or_task_name = %MIXTURE_NAME" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"

    --gin_file="perplexity_eval.gin" \

# IndicTrans FT
export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=eval
export TPU_SIZE=v3-8
export TASK=baseline_hi
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="gs://pre-train/transliterated/fine-tune"

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
  --gin_file="gs://pre-train/transliterated/og-model-dir/operative_config.gin" \
  --module_import="multilingual_t5.tasks"