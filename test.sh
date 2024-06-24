cd mapper
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
--exp_name="released_version" \
--description="released_version" \
--output_dir='../outputs_test' \
--test \
# Set data path below in train_options.py
# self.parser.add_argument("--test", default=True) #, action="store_true"

# --test_data_list='C:\VITON\FashionTex_Generate\data\test_data\test_example.json' \
# --test_img_dir='C:\\VITON\\FashionTex_Generate\\img' \
# --test_texture_dir='C:\\VITON\\FashionTex_Generate\\texture' \
# --checkpoint_path='C:\\VITON\\FashionTex_Generate\\version_5\\checkpoints\\epoch=593-step=1672703.ckpt' \


# --test_data_list='C:\\VITON\\FashionTex_Generate\\data\\test_data\\output.json' \
# --test_img_dir='C:\\VITON\\FashionTex_Generate\\data\\test_data\\img_test' \
# --test_texture_dir='C:\\VITON\\FashionTex_Generate\\data\\test_data\\texture_test' \
# --checkpoint_path='C:\\VITON\\FashionTex_Generate\\version_5\\checkpoints\\last.ckpt' \
