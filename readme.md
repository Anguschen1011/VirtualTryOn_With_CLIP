# A Reproducing and fine-tuning version of FashionTex

## 1. Environment
```conda env create -f environment.yaml  ```

## 2. Data Preprocessing (Steps for handling and installation of relevant packages : StyleGAN-Human [[Github]](https://github.com/stylegan-human/StyleGAN-Human) )  
```cd process_data  ```

#### 2.1 Aligned raw images [[Github]](https://github.com/stylegan-human/StyleGAN-Human/tree/main#aligned-raw-images)  
```python alignment.py --image-folder img/test/ --output-folder aligned_image/  ```

#### 2.2 Invert real image with PTI [[Github]](https://github.com/stylegan-human/StyleGAN-Human/tree/main?tab=readme-ov-file#invert-real-image-with-pti)  
```
python run_pti.py
cd..
```

Before inversion, please download our PTI weights: e4e_w+.pt into ```/pti/.```  

Few parameters you can change:  
```/pti/pti_configs/hyperparameters.py```:  
first_inv_type = ```'w+'``` -> Use pretrained e4e encoder  
first_inv_type = ```'w'``` -> Use projection and optimization  
```/pti/pti_configs/paths_config.py```:  
input_data_path: path of real images  
```e4e```: path of e4e_w+.pt  
```stylegan2_ada_shhq```: pretrained stylegan2-ada model for SHHQ  

we only need the output embedding "0.pt" in ```'outputs/pti/'```.  
Since only need the output of e4e, you can comment out the finetuning code to save time.  

## 3. Training
```bash run.sh```  
 
You can set the GPU number in run.sh  
If you would like to change the data, weights, output path or other settings,   
you can find them in ```mapper/options/train_options.py```.  
 

## 4. Testing  

#### 4.1 Step 1  
First set the 'checkpoint_path','test_data_list', 'test_img_dir' and 'test_texture_dir' in ```mapper/options/train_options.py```. 
```bash test.sh``` 
Download our pretrained weight from : [[LINK]](https://mega.nz/file/eYE2UbbQ#Oti_jfYjcF_WqmcLxrcAEjMmAoM_ZquhRZWEmQmfoeU)

#### 4.2 Step  2 (ID Recovery Module)  
Set the data path in ```'recovery_module/pti/pti_configs/paths_config.py'```.  
```
cd recovery_module  
python run_pti.py
``` 

![result_example](results/Img_1.png)
<p align="center"><i>Figure 1: Result Image.</i></p>  

![result_example](results/report.png)
<p align="center"><i></i></p>

## 5. Hardware
The model architectures proposed in this study are implemented using the PyTorchDL framework, and training is conducted on hardware featuring an ```Intel® Core™ i7-12700``` CPU and ```Nvidia RTX 3060``` graphics processing unit (GPU).

# References： 
> * FashionTex: Controllable Virtual Try-on with Text and Texture  
>[[FashionTex]](https://github.com/picksh/FashionTex)  
> * StyleGAN-Human: A Data-Centric Odyssey of Human Generation  
>[[StyleGAN-Human]](https://github.com/stylegan-human/StyleGAN-Human)

