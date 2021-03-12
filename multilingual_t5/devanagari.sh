export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://pre-train
export TPU_NAME=fine-tune
export TPU_SIZE=v3-8
export DATA_DIR="${BUCKET}/transliterated/data"
export MODEL_DIR="${BUCKET}/transliterated/model-dir"
export TASK=devanagari

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}/tfds" \
  --project_name="devanagari" \
  --gin_file="models/t5.1.1.base.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 131072)" \
  --gin_param="run.train_steps = 500000" \
  --gin_param="run.save_checkpoints_steps = 100000" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.learning_rate_schedule = @learning_rate_schedules.learning_rate_schedule_noam" \
  --gin_param="learning_rate_schedule_noam.linear_decay_fraction = 0.0" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.mesh_train_dataset_fn" \
  --gin_param="mesh_train_dataset_fn.mixture_or_task_name = %MIXTURE_NAME" \
  --gin_param="mesh_transformer.get_vocabulary.mixture_or_task_name = %MIXTURE_NAME" \
  --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
  --module_import="multilingual_t5.tasks"