# Reproducing and fine-tuning FashionTex [FashionTex](https://github.com/picksh/FashionTex)

## 1. Environment
conda env create -f environment.yaml  

## 2. Data Preprocessing (Steps for handling and installation of relevant packages : StyleGAN-Human [[Github]](https://github.com/stylegan-human/StyleGAN-Human) )  
cd process_data  

#### 2.1 Aligned raw images (GitHub:https://github.com/stylegan-human/StyleGAN-Human/tree/main#aligned-raw-images)  
python alignment.py --image-folder img/test/ --output-folder aligned_image/  

#### 2.2 Invert real image with PTI (GitHub:https://github.com/stylegan-human/StyleGAN-Human/tree/main?tab=readme-ov-file#invert-real-image-with-pti)  
python run_pti.py  
cd..  
"""
Before inversion, please download our PTI weights: e4e_w+.pt into /pti/.  

Few parameters you can change:  

/pti/pti_configs/hyperparameters.py:  
first_inv_type = 'w+' -> Use pretrained e4e encoder  
first_inv_type = 'w' -> Use projection and optimization  
/pti/pti_configs/paths_config.py:  
input_data_path: path of real images  
e4e: path of e4e_w+.pt  
stylegan2_ada_shhq: pretrained stylegan2-ada model for SHHQ  

we only need the output embedding "0.pt" in 'outputs/pti/'.  
Since we only need the output of e4e, you can comment out the finetuning code to save time.  
"""  

## 3. Training
bash run.sh  
"""  
You can set the GPU number in run.sh  
If you would like to change the data, weights, output path or other settings,   
you can find them in mapper/options/train_options.py.  
"""  

## 4. Testing  

#### 4.1 Step 1  
First set the 'checkpoint_path','test_data_list', 'test_img_dir' and 'test_texture_dir' in mapper/options/train_options.py.  
bash test.sh  

#### 4.2 Step  2 (ID Recovery Module)  
Set the data path in 'recovery_module/pti/pti_configs/paths_config.py'.  
cd recovery_module  
python run_pti.py  
