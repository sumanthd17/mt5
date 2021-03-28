export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=eval2
export TPU_SIZE=v3-8
export LANG=pa
export TASK=baseline_pa
export CKPT=1045000
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-model-dir"

for SET in all wat2021-devtest
do 
    python -m t5.models.mesh_transformer_main \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="${MODEL_DIR}/config.gin" \
    --gin_file="infer.gin" \
    --gin_file="beam_search.gin" \
    --gin_param="Bitransformer.decode.beam_size = 5" \
    --gin_param="input_filename = '${BUCKET}/baselines/mT5/devtest/${SET}/en-${LANG}/test.${LANG}'"\
    --gin_param="output_filename = '${BUCKET}/baselines/mT5/forward/mT5-${SET}-${LANG}-en'"\
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"\
    --gin_param="infer_checkpoint_step = ${CKPT}" \
    --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
    --module_import="multilingual_t5.tasks"

    echo "Finished $SET"
done