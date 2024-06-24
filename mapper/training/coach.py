import os
import json
import torch
import pickle
import numpy as np
import torchvision
from torch import nn
from PIL import Image
from criteria import id_loss
import pytorch_lightning as pl
import torch.nn.functional as F
import criteria.clip_loss as clip_loss
from criteria.human_parse import seg_loss
from mapper.clip_mapper import CLIPMapper
from mapper.training.ranger import Ranger
from torch.utils.tensorboard import SummaryWriter
## Changed
import lpips
import clip
from criteria.human_parse.style_loss import StyleLoss
from criteria.human_parse.denseclip.denseclip import DenseCLIP
from criteria.human_parse.configs.denseclip_fpn_vit_b_640x640_80k import CONF,data_meta
from models.facial_recognition.model_irse import Backbone
from mapper.torch_utils.models import Generator

import warnings
warnings.filterwarnings('ignore')

class Coach(pl.LightningModule):
	def __init__(self, opts):
		super().__init__()
		self.save_hyperparameters()
		self.opts = opts
		
  		# Loading CLIP Model
		print('Loading CLIP Loss')
		self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
		self.clip_model.eval()
		# print("CLIP Model in Coach.py : ", id(self.clip_model))
		self.upsample = torch.nn.Upsample(scale_factor=7)
		self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
		self.cos_loss2= torch.nn.CosineEmbeddingLoss()

		# Loading Seg_model
		print('Loading SegLoss from DenseCLIP')
		seg_model_name='clip'
		self.seg_model_name=seg_model_name
		self.style_loss=StyleLoss()
		if seg_model_name=='clip':
			self.seg_model=DenseCLIP(**CONF)
			self.seg_model.load_state_dict(torch.load(opts.seg_model_path)['state_dict'])
			self.seg_model.cuda()
			self.seg_model.eval()
		self.bg_mask_l2_loss = nn.MSELoss()
		self.color_l1_loss = nn.L1Loss()
		self.percept = lpips.LPIPS(net='vgg').cuda()
		self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])

		# Loading ID loss
		print('Loading ResNet ArcFace for ID Loss')
		self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
		self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
		self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
		self.facenet.eval()
  
		# Initialize network
		self.net = CLIPMapper(self.opts, self.clip_model).to(self.device)
		
  		# Initialize loss
		# self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
		# self.clip_loss = clip_loss.CLIPLoss(opts, self.openclip_model)
		self.latent_l2_loss = nn.MSELoss().to(self.device).eval()
		# self.seg_loss=seg_loss.SegLoss(opts,seg_model=opts.seg_model)

		# Initialize logger
		log_dir = os.path.join(opts.output_dir,opts.exp_name,'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir

		if not self.opts.test:
			opts_dict = vars(opts)
			with open(os.path.join(opts.output_dir, opts.exp_name,'opt.json'), 'w') as f:
				json.dump(opts_dict, f, indent=4, sort_keys=True)
		
		if opts.optim_name=='ranger':
			self.automatic_optimization = False

	def forward(self,w, type_text_emb_up, color_tensor_up, color_tensor_low, type_text_emb_low):
		
		w_hat=w+0.1*self.net.mapper(w, type_text_emb_up, color_tensor_up, color_tensor_low, type_text_emb_low)
		x_hat, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)

		return x_hat, w_hat

	def _shared_eval(self, batch, batch_idx, prefix):
		
		w,  type_text_emb_up, type_text_emb_low, selected_description, color_tensor_up,color_tensor_low, target_type, image_label= batch
		with torch.no_grad():
			x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
		x_hat, w_hat=self(w, type_text_emb_up, color_tensor_up, color_tensor_low, type_text_emb_low)

		if self.opts.test:
			return x,x_hat,color_tensor_up,color_tensor_low,selected_description, image_label, w_hat

		loss, loss_dict = self.calc_loss(w, x, w_hat, x_hat, color_tensor_up,color_tensor_low, target_type, image_label, log_seg=False, prefix=prefix)
		self.log_dict(loss_dict,on_step=(prefix=="train"), on_epoch=True, prog_bar=True)
		
		return loss, loss_dict, x, x_hat, color_tensor_up,color_tensor_low, selected_description

	def training_step(self,batch,batch_idx):
		if self.opts.optim_name=='ranger':
			optimizer = self.optimizers()
			optimizer.zero_grad()
		
		loss, loss_dict, x, x_hat, color_tensor_up,color_tensor_low, selected_description=self._shared_eval(batch,batch_idx,prefix="train")
		
		if self.opts.optim_name=='ranger':
			self.manual_backward(loss)
			optimizer.step()
		
		if self.global_step % self.opts.image_interval == 0 or (
				self.global_step < 1000 and self.global_step % 1000 == 0):
			self.log_loss(loss_dict)
			self.parse_and_log_images(x, x_hat, color_tensor_up,color_tensor_low, title='images_train', selected_description=selected_description)

		return loss
		
	def validation_step(self,batch,batch_idx):
		
		loss, loss_dict, x, x_hat, color_tensor_up,color_tensor_low,selected_description=self._shared_eval(batch,batch_idx,prefix="test")
		
		if batch_idx%100==0:
			self.parse_and_log_images(x, x_hat, color_tensor_up,color_tensor_low ,title='images_val', selected_description=selected_description, index=batch_idx)
			self.log_loss(loss_dict)
		
		return loss

	def test_step(self, batch, batch_idx):
		x,x_hat,color_tensor_up,color_tensor_low,selected_description, image_name, w_hat=self._shared_eval(batch, batch_idx, prefix='test')
		#x,x_hat,color_tensor,color_tensor2,selected_description, image_name, w_hat=self._shared_eval(batch, batch_idx, prefix='test')
		# Only save final results
		self.log_image(x_hat, image_name, selected_description, title='images_test/img')
		# Use self.parse_and_log_images to save original img, output, and ref textrues.
		
		# Save w_hat for recovery module
		self.log_w(w_hat, image_name, selected_description, title='images_test/w')

	def configure_optimizers(self):	
		
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(self.net.mapper.parameters(), lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(list(self.net.mapper.parameters()), lr=self.opts.learning_rate)
		return optimizer

	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
		optimizer.step(closure=optimizer_closure)

	def calc_loss(self, w, x, w_hat, x_hat, color_tensor_up, color_tensor_low, target_type, image_label, log_seg=False,prefix='',real_image=None,des=None):
		loss_dict = {}
		loss = 0.0
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement = self.id_loss_forward(x_hat, x)
			loss_dict[f"{prefix}_loss_id"] = float(loss_id* self.opts.id_lambda * self.opts.attribute_preservation_lambda)
			loss = loss_id * self.opts.id_lambda * self.opts.attribute_preservation_lambda
		
		if self.opts.latent_l2_lambda > 0:
			loss_l2_latent = self.latent_l2_loss(w_hat, w)
			loss_dict[f"{prefix}_loss_l2_latent"] = float(loss_l2_latent* self.opts.latent_l2_lambda * self.opts.attribute_preservation_lambda)
			loss += loss_l2_latent * self.opts.latent_l2_lambda * self.opts.attribute_preservation_lambda
		
		if self.opts.skin_lambda > 0:
			loss_skin=self.loss_skin_only(x_hat, x)
			
			if self.opts.background_lambda>0:
				loss_bg=loss_skin[1]
				loss_skin=loss_skin[0]
				
				loss_dict[f"{prefix}_loss_bg"]=float(loss_bg*self.opts.background_lambda*self.opts.attribute_preservation_lambda)
				loss+=loss_bg*self.opts.background_lambda*self.opts.attribute_preservation_lambda

			loss_dict[f"{prefix}_loss_skin"]=float(loss_skin*self.opts.skin_lambda*self.opts.attribute_preservation_lambda)
			loss+=loss_skin*self.opts.skin_lambda*self.opts.attribute_preservation_lambda
		
		loss_perceptual1=self.perceptual_loss_only(color_tensor_up,x_hat,part="upper")
		loss_perceptual2=self.perceptual_loss_only(color_tensor_low,x_hat,part="lower")
		loss_perceptual=(loss_perceptual1+loss_perceptual2)/2

		if self.opts.image_color_lambda>0:
			loss_color1=self.image_color_loss_only(color_tensor_up,x_hat,part="upper")
			loss_color2=self.image_color_loss_only(color_tensor_low,x_hat,part="lower")
			loss_color=(loss_color1+loss_color2)/2
			loss_dict[f"{prefix}_loss_image_color"]=float(loss_color*self.opts.image_color_lambda)
			loss+= loss_color*self.opts.image_color_lambda
			
		loss_dict[f"{prefix}_loss_perceptual"]=float(loss_perceptual*self.opts.perceptual_lambda*self.opts.image_manipulation_lambda)
		loss+= loss_perceptual*self.opts.perceptual_lambda*self.opts.image_manipulation_lambda
			
		loss_type=self.new_clip_loss(x,image_label,x_hat,target_type)
		# loss_type=self.clip_loss.new_clip_loss(x,image_label,x_hat,target_type)
		loss_dict[f"{prefix}_loss_type"] = float(loss_type* self.opts.text_manipulation_lambda)
		loss += loss_type * self.opts.text_manipulation_lambda
		
		loss_dict[f"{prefix}_loss"] = float(loss)
		
		return loss, loss_dict

	def parse_and_log_images(self, img, img_hat, color_tensor ,color_tensor2, title, selected_description, index=None, real_image=None):
		x=img[0].unsqueeze(0).detach().cpu()
		x_hat=img_hat[0].unsqueeze(0).detach().cpu()
		img_tensor=color_tensor[0].unsqueeze(0).detach().cpu()
		img_tensor2=color_tensor2[0].unsqueeze(0).detach().cpu()
		selected_description=selected_description[0]
		if self.opts.test:
			index=index[0]
		
		if index is None:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}-{selected_description}.jpg')
		else:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}-{str(index).zfill(5)}-{selected_description}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		
		color_tensor_pad=(int((x.shape[3]-img_tensor.shape[3])/2),int((x.shape[3]-img_tensor.shape[3])/2),int((x.shape[2]-img_tensor.shape[2])/2),int((x.shape[2]-img_tensor.shape[2])/2))
		torchvision.utils.save_image(torch.cat([x, x_hat, F.pad(img_tensor,pad=color_tensor_pad),F.pad(img_tensor2,pad=color_tensor_pad)]), path,
								     normalize=True, scale_each=True, value_range=(-1, 1), nrow=4)
		# torchvision.utils.save_image(torch.cat([x, x_hat, F.pad(img_tensor,pad=color_tensor_pad),F.pad(img_tensor2,pad=color_tensor_pad)]), path, normalize=True, scale_each=True, range=(-1, 1), nrow=4)
	# TODO: May Need modify according to different output
	def log_image(self,x_hat,image_name,des, title):
		image_name=image_name[0].split('.')[0]
		des=des[0]
		path=os.path.join(self.log_dir,title, f'{image_name}+{des}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torchvision.utils.save_image(torch.cat([x_hat.detach().cpu()]),path,normalize=True, scale_each=True, value_range=(-1, 1))
  		# torchvision.utils.save_image(torch.cat([x_hat.detach().cpu()]),path,normalize=True, scale_each=True, range=(-1, 1))
	# TODO: May Need modify according to different output
	def log_w(self, w, image_name, des, title):
		image_name=image_name[0].split('.')[0]
		des=des[0]
		path=os.path.join(self.log_dir,title, f'{image_name}+{des}.npy')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		np.save(path, w.cpu().numpy())

		
	def log_loss(self, loss_dict):
		with open(os.path.join(self.log_dir, 'timestamp.txt'), 'a') as f:
			f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))
	
	## CLIP Loss
	@torch.no_grad()
	def new_clip_loss(self, ori_image,ori_text,tar_image,tar_text):
		
		ori_image = self.avg_pool(self.upsample(ori_image))
		tar_image = self.avg_pool(self.upsample(tar_image))

		emb_ori=self.clip_model.encode_image(F.interpolate(ori_image,size=224))
		emb_tar=self.clip_model.encode_image(F.interpolate(tar_image,size=224))

		emb_ori_text=self.clip_model.encode_text(clip.tokenize(ori_text[0]).cuda())
		emb_tar_text=self.clip_model.encode_text(clip.tokenize(tar_text[0]).cuda())
		change_text=torch.from_numpy(np.array(ori_text[1]==ori_text[0])).view(-1,1).repeat(1,emb_ori_text.shape[1]).cuda()
		emb_ori_text2=self.clip_model.encode_text(clip.tokenize(ori_text[1]).cuda())
		emb_ori_text2=torch.where(change_text==False,torch.zeros_like(emb_ori_text),emb_ori_text2)
		change_text=torch.from_numpy(np.array(tar_text[1])==np.array(tar_text[0])).view(-1,1).repeat(1,emb_tar_text.shape[1]).cuda()
		emb_tar_text2=self.clip_model.encode_text(clip.tokenize(tar_text[1]).cuda())
		emb_tar_text2=torch.where(change_text==False,torch.zeros_like(emb_tar_text),emb_tar_text2)
		t_res=emb_ori-emb_ori_text-emb_ori_text2
		t_full=emb_tar_text+emb_tar_text2+t_res

		cos_target = torch.ones((emb_tar.shape[0])).float().cuda()
		similarity = self.cos_loss2(emb_tar, t_full, cos_target)
		return similarity
	
	### Seg Loss
	# cal lab written by liuqk
	def f(self, input):
		output = input * 1
		mask = input > 0.008856
		output[mask] = torch.pow(input[mask], 1 / 3)
		output[~mask] = 7.787 * input[~mask] + 0.137931
		return output

	def rgb2xyz(self, input):
		assert input.size(1) == 3
		M_tmp = self.M.to(input.device).unsqueeze(0)
		M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC
		output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW
		M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
		M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1
		return output / M_tmp

	def xyz2lab(self, input):
		assert input.size(1) == 3
		output = input * 1
		xyz_f = self.f(input)
		# compute l
		mask = input[:, 1, :, :] > 0.008856
		output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
		output[:, 0, :, :][~mask] = 903.3 * input[:, 1, :, :][~mask]
		# compute a
		output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])
		# compute b
		output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])
		return output

	def seg_with_text(self,seg_result,text,seg_otherwise=False):

		mask=torch.zeros_like(seg_result)
		#if 'outer' in text:
		#    text.append('top')
		if 'top' in text and 'outer' not in text:
			text.insert(0,'outer')
		for class_name in text:
			choose_text=class_name
			if self.seg_model_name=='clip':
				class_id=CONF['class_names'].index(class_name)
			else:
				class_id=self.model.label.index(class_name)
			mask=torch.where(seg_result==class_id,torch.ones_like(mask),mask) # 1 1024 512
			if mask.max()==1:
				break
		if seg_otherwise:
			otherwise_mask=torch.zeros_like(seg_result)
			otherwise_mask=torch.where(seg_result!=class_id ,torch.ones_like(otherwise_mask),otherwise_mask) # 1 1024 512
			otherwise_mask=torch.where(seg_result!=0,torch.ones_like(otherwise_mask),otherwise_mask)
			return otherwise_mask.unsqueeze(1)

		return mask.unsqueeze(1),choose_text#,class_id

	def seg_bg(self,seg_result):
		cloth=[1,2,3,4,5,6,21]
		mask=torch.ones_like(seg_result)
		
		for i in cloth:
			mask=torch.where(seg_result==i,torch.zeros_like(mask),mask) # 1 1024 512
		return mask.unsqueeze(1)

	def cal_avg(self,input,mask):
		#print('mask',mask.shape)
		#print('input',input.shape)
		x = input * mask
		sum = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True) # [n,3,1,1]
		mask_sum = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True) # [n,1,1,1]
		mask_sum[mask_sum == 0] = 1
		avg = sum / mask_sum
		
		# print(avg)
		#print(t)
		return avg
	def crop_image(self,img,sample_size,part="upper"):
		cropped=torch.zeros(img.shape[0],sample_size,sample_size,3).cuda()
		
		#upper
		if part=="upper":
			i=257
			j=200
		#lower
		elif part=="lower":
			i=515
			j=187
		cropped=img[:,i:i+sample_size,j:j+sample_size]

		return cropped

	def image_color_loss_only(self,patch,x_hat,part="upper"):
		mask_fake=torch.ones_like(patch)
		mask_real=torch.ones_like(patch)
		img_tensor=patch
		gen_crop=self.crop_image(x_hat.permute(0,2,3,1),img_tensor.shape[-1],part=part) # B 64 64 3
		gen_crop=gen_crop.permute(0,3,1,2) # 1 3 64 64
		color_loss=self.calc_color_loss(img_tensor,gen_crop,mask_real,mask_fake)
		return color_loss

	def loss_image_texture(self,patch,x_hat,part="upper"):
		
		mask_fake=torch.ones_like(patch)
		mask_real=torch.ones_like(patch)
		
		#perceptual loss
		img_tensor=patch
		gen_crop=self.crop_image(x_hat.squeeze().permute(1,2,0),img_tensor.shape[-1],part=part) # 1 64 64 3
		gen_crop=gen_crop.permute(0,3,1,2) # 1 3 64 64
		
		percept_loss = 250*self.style_loss(img_tensor,gen_crop).mean()
		
		color_loss=self.calc_color_loss(img_tensor,gen_crop,mask_real,mask_fake)
		
		return color_loss,percept_loss

	def perceptual_loss_only(self,fake,real,part="upper"):
		#perceptual loss
		img_tensor=fake
		gen_crop=self.crop_image(real.permute(0,2,3,1),img_tensor.shape[-1],part=part) # B 64 64 3
		gen_crop=gen_crop.permute(0,3,1,2) # 1 3 64 64
		
		percept_loss = 250*self.style_loss(img_tensor,gen_crop).mean()
		#color_loss=0.
		if self.opts.texture_loss_type=='clip':
			return gen_crop
		else:
			return percept_loss


	def calc_color_loss(self,x,x_hat,mask_x,mask_x_hat):
		x_hat_RGB=(x_hat+1)/2.0
		x_RGB=(x+1)/2.0

		# from RGB to Lab by liuqk
		x_xyz = self.rgb2xyz(x_RGB)
		x_Lab = self.xyz2lab(x_xyz)
		x_hat_xyz = self.rgb2xyz(x_hat_RGB)
		x_hat_Lab = self.xyz2lab(x_hat_xyz)

		# cal average value
		x_Lab_avg = self.cal_avg(x_Lab, mask_x)
		x_hat_Lab_avg = self.cal_avg(x_hat_Lab, mask_x_hat)

		color_loss = self.color_l1_loss(x_Lab_avg, x_hat_Lab_avg)

		return color_loss

	def loss_skin_only(self,x,x_hat):
		seg_x, seg_x_hat= self.get_seg_only(x,x_hat)
		#if self.seg_model=='clip':
		skin_text=['skin']
		mask_x,_=self.seg_with_text(seg_x,skin_text)
		mask_x_hat, _=self.seg_with_text(seg_x_hat,skin_text)
		color_loss=self.calc_color_loss(x,x_hat,mask_x,mask_x_hat)
		if self.opts.background_lambda>0:
			mask_x_bg=self.seg_bg(seg_x)
			mask_x_hat_bg=self.seg_bg(seg_x_hat)
			mask=mask_x_bg*mask_x_hat_bg
			bg_loss=self.bg_mask_l2_loss(x*mask, x_hat*mask)
			return (color_loss,bg_loss)
		return color_loss

	def get_seg_only(self,x,x_hat):
		if self.seg_model_name=='clip':
			skin_text=['skin']
			'''
			CLASSES = ('background', 'top','outer','skirt','dress','pants','leggings','headwear',
					'eyeglass','neckwear','belt','footwear','bag','hair','face','skin',
					'ring','wrist wearing','socks','gloves','necklace','rompers','earrings','tie')
			'''
			with torch.no_grad():
				seg_result_x=self.seg_model.simple_test(x,data_meta) # 1 1024 512
				seg_result_x_hat=self.seg_model.simple_test(x_hat,data_meta) # 1 1024 512
			
		else:
			skin_text=['Torso']
			seg_result_x=self.seg_model.get_seg(x.squeeze().permute(1,2,0).detach().cpu().numpy())
			seg_result_x_hat=self.seg_model.get_seg(x_hat.squeeze().permute(1,2,0).detach().cpu().numpy())

		seg_result_x=torch.from_numpy(np.array(seg_result_x)).cuda()
		seg_result_x_hat=torch.from_numpy(np.array(seg_result_x_hat)).cuda()

		return seg_result_x,seg_result_x_hat
	
	def extract_feats(self, x):
		if x.shape[2] != 256:
			x = self.pool(x)
		x = x[:, :, 35:223, 32:220]  # Crop interesting region
		x = self.face_pool(x)
		#sprint(x.shape)
		x_feats = self.facenet(x)
		return x_feats

	def id_loss_forward(self, y_hat, y):
		n_samples = y.shape[0]
		y_feats = self.extract_feats(y)  # Otherwise use the feature from there
		y_hat_feats = self.extract_feats(y_hat)
		y_feats = y_feats.detach()
		loss = 0
		sim_improvement = 0
		count = 0
		for i in range(n_samples):
			diff_target = y_hat_feats[i].dot(y_feats[i])
			loss += 1 - diff_target
			count += 1

		return loss / count, sim_improvement / count