export PROJECT=ai4b-word-embeddings
export ZONE=europe-west4-a
export BUCKET=gs://ai4b-anuvaad-nmt
export TPU_NAME=eval4
export TPU_SIZE=v3-8
export LANG=kn
export TASK=baseline_kn
export CKPT=1030600
export MODEL_DIR="${BUCKET}/baselines/mT5/${TASK}/og-r-model-dir"

for SET in all anuvaad-legal wat2021-devtest
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
    --gin_param="input_filename = '${BUCKET}/baselines/mT5/devtest/${SET}/en-${LANG}/test.en'"\
    --gin_param="output_filename = '${BUCKET}/baselines/mT5/reverse/mT5-${SET}-en-${LANG}'"\
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"\
    --gin_param="infer_checkpoint_step = ${CKPT}" \
    --gin_param="utils.run.vocabulary = @mesh_transformer.get_vocabulary()" \
    --module_import="multilingual_t5.tasks"

    echo "Finished $SET"
done