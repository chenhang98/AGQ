export CUDA_VISIBLE_DEVICES=0

IMAGE_FILE=datasets/AC3-AC4/AC3_inputs.h5
LABLE_FILE=datasets/AC3-AC4/AC3_labels.h5

OUTPUT_DIR=outputs/inference
PKL_PATH=${OUTPUT_DIR}/pkl_results
CKPT_FILE=./checkpoints/checkpoint_200000.pth.tar


if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi


# ------------ Blockwise inference and evaluation ------------ #
python scripts/inference_mp.py \
    --config-base projects/AGQ/configs/SNEMI-Base.yaml \
    --config-file projects/AGQ/configs/AGQ.yaml \
    --inference \
    --checkpoint ${CKPT_FILE} \
    SYSTEM.PARALLEL None \
    INFERENCE.STRIDE '[16,256,256]' \
    INFERENCE.IMAGE_NAME ${IMAGE_FILE} \
    INFERENCE.LABEL_PATH ${LABLE_FILE} \
    INFERENCE.OUTPUT_PATH ${PKL_PATH} \
    MODEL.INIT_MASK_METHOD connected_components \
    MODEL.INFERENCE_WITHOUT_BG True 


# ------------ Full volume concat, merge and evaluation ------------ #
python scripts/concat_merge_eval.py \
    --pkl_path ${PKL_PATH} \
    --image_path ${IMAGE_FILE} \
    --label_path ${LABLE_FILE} \
    --out_path ${OUTPUT_DIR} 