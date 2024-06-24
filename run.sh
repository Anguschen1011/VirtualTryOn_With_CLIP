cd mapper
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
--exp_name="released_version" \
--description="released_version" \
--text_manipulation_lambda=80 \
--id_lambda=0.1 \
--latent_l2_lambda=0.8 \
--background_lambda=100 \
--skin_lambda=0.1 \
--perceptual_lambda=8 \
--image_color_lambda=1.0 \
--batch_size=4 \
# python -m cProfile -o data.pstats -s cumtime scripts/train.py
# python scripts/train.py