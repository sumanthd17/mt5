export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export LANG=bn
export TPU_NAME=r-ic-all-${LANG}
export TPU_SIZE=v3-8
export TASK=r_ic_all_${LANG}
export DATA_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/tfds"
export MODEL_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/${TASK}"

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


export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export LANG=bn
export TPU_NAME=r-indic-corp-${LANG}
export TPU_SIZE=v2-8
export TASK=r_indic_corp_${LANG}
export DATA_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/tfds"
export MODEL_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/${TASK}"

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
  --gin_param="run.train_steps = 1131072" \
  --gin_param="run.save_checkpoints_steps = 5000" \
  --gin_param="learning_rate_schedules.constant_learning_rate.learning_rate = 0.001" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 65536)" \
  --gin_file="gs://t5-data/pretrained_models/mt5/base/operative_config.gin" \
  --module_import="multilingual_t5.tasks"


# eval
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=mt5-eval-2
export TPU_SIZE=v2-8
export LANG='hi'
export TASK=r_ic_all_${LANG}
export DATA_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/tfds"
export MODEL_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/${TASK}"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"


export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=mt5-eval-3
export TPU_SIZE=v2-8
export LANG='ta'
export TASK=r_ic_all_${LANG}
export DATA_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/tfds"
export MODEL_DIR="${BUCKET}/ai4b-models/mT5/${LANG}/${TASK}"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"