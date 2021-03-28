# bn
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=bn-r
export TPU_SIZE=v3-8
export TASK=baseline_bn
export MIXTURE=r_baseline_bn
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
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

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"

# hi
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=hi
export TPU_SIZE=v3-8
export TASK=baseline_hi
export MIXTURE=r_baseline_hi
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"


# kn
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-kn
export TPU_SIZE=v2-8
export TASK=baseline_kn
export MIXTURE=r_baseline_kn
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"

# ml
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-ml
export TPU_SIZE=v2-8
export TASK=baseline_ml
export MIXTURE=r_baseline_ml
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
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

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"

# pa
export PROJECT=ai4b-word-embeddings
export ZONE=us-central1-f
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=ssd-r-mt5-pa
export TPU_SIZE=v2-8
export TASK=baseline_pa
export MIXTURE=r_baseline_pa
export DATA_DIR="${BUCKET}/baselines/mT5/${TASK}/tfds"
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
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

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
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

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${MIXTURE}'" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"