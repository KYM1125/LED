
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn

from utils.config import Config
from utils.utils import print_log


from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate
import matplotlib.pyplot as plt
from IPython.display import Image, display
from torch.utils.tensorboard import SummaryWriter

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from models.model_led_initializer_with_intention import VecIntInitializer as MyInitializationModel

import pdb
NUM_Tau = 5
writer = SummaryWriter(log_dir="runs/angle_comparison")
writer_loss = SummaryWriter(log_dir='runs/loss')


class Trainer:
	def __init__(self, config):
		
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
		self.cfg = Config(config.cfg, config.info)
		
		# ------------------------- prepare train/test data loader -------------------------
		train_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=True)

		self.train_loader = DataLoader(
			train_dset,
			batch_size=self.cfg.train_batch_size,
			shuffle=True,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)
		
		test_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=False)

		self.test_loader = DataLoader(
			test_dset,
			batch_size=self.cfg.test_batch_size,
			shuffle=False,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)
		
		# data normalization parameters
		self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
		self.traj_scale = self.cfg.traj_scale

		# ------------------------- define diffusion parameters -------------------------
		self.n_steps = self.cfg.diffusion.steps # define total diffusion steps

		# make beta schedule and calculate the parameters used in denoising process.
		'''
			schedule: 控制噪声调度的方式（如线性、cosine等）。
			n_timesteps: 时间步数，表示整个扩散过程分为多少个步骤。
			start: 在扩散的开始时，噪声强度的起始值。
			end: 在扩散的结束时，噪声强度的结束值。
		'''
		self.betas = self.make_beta_schedule(
			schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps, 
			start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).cuda()
		
		self.alphas = 1 - self.betas
		self.alphas_prod = torch.cumprod(self.alphas, 0)
		self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
		self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)


		# ------------------------- define models -------------------------
		self.model = CoreDenoisingModel().cuda()
		# load pretrained models
		model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu')
		self.model.load_state_dict(model_cp['model_dict'])

		self.model_initializer = MyInitializationModel(t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20).cuda()

		self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
		self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)
		
		# ------------------------- prepare logs -------------------------
		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		self.print_model_param(self.model, name='Core Denoising Model')
		self.print_model_param(self.model_initializer, name='Initialization Model')

		# temporal reweight in the loss, it is not necessary.
		self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10


	def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
		'''
		Count the trainable/total parameters in `model`.
		'''
		total_num = sum(p.numel() for p in model.parameters())
		trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print_log("[{}] Trainable/Total: {}/{}".format(name, trainable_num, total_num), self.log)
		return None
	

	def make_beta_schedule(self, schedule: str = 'linear', 
			n_timesteps: int = 1000, 
			start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
		'''
		Make beta schedule.

		Parameters
		----
		schedule: str, in ['linear', 'quad', 'sigmoid'],
		n_timesteps: int, diffusion steps,
		start: float, beta start, `start<end`,
		end: float, beta end,

		Returns
		----
		betas: Tensor with the shape of (n_timesteps)

		'''
		if schedule == 'linear':
			betas = torch.linspace(start, end, n_timesteps)
		elif schedule == "quad":
			betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
		elif schedule == "sigmoid":
			betas = torch.linspace(-6, 6, n_timesteps)
			betas = torch.sigmoid(betas) * (end - start) + start
		return betas


	def extract(self, input, t, x):
		shape = x.shape
		out = torch.gather(input, 0, t.to(input.device))
		reshape = [t.shape[0]] + [1] * (len(shape) - 1)
		return out.reshape(*reshape)

	def noise_estimation_loss(self, x, y_0, mask):
		'''
			x是语义信息（未来轨迹的社交信息，或者是过去轨迹），y_0代表了无噪声的原始值（未来轨迹），mask是一个掩码张量，用于指示哪些值是有效的。
		'''
		batch_size = x.shape[0]
		# Select a random step for each example
		t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device) # 随机选择一个时间步，通过这些时间步，模型可以学习如何去估计或去除不同时间步的噪声。
		t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
		'''
			self.alphas_bar_sqrt是一个递减的序列，表示每一步去噪过程中，噪声的强度逐渐减小。
			self.betas是一个递增的序列，表示每一步扩散过程中，噪声的强度逐渐增大的过程。
			one_minus_alphas_bar_sqrt是alphas_bar_sqrt的补集,反映了剩余的噪声强度。
			self.extract函数是一个辅助函数，从这些系数张量中提取出与当前时间步 t 对应的值。
		'''
		# x0 multiplier
		a = self.extract(self.alphas_bar_sqrt, t, y_0)
		beta = self.extract(self.betas, t, y_0)
		# eps multiplier
		am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
		e = torch.randn_like(y_0)
		# model input
		y = y_0 * a + e * am1 # 将噪声加入到无噪的原始值上
		output = self.model(y, beta, x, mask)# 通过模型去估计噪声
		# batch_size, 20, 2
		return (e - output).square().mean()

	def p_sample(self, x, mask, cur_y, t): # 没有加速的去噪过程
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model(cur_y, beta, x, mask)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z
		return (sample)
	
	def p_sample_accelerate(self, x, mask, cur_y, t): # 加速后的去噪过程
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y)) # 去噪强度
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y) # beta 参数控制在每一步扩散过程中，噪声加入的程度。
		eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)# cur_y: [22,10,20,2], beta: [22,1,1,1], x: [22,10,6], mask: [22,22] # 预测的噪声
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta)) # 去噪后的均值
		# Generate z
		z = torch.randn_like(cur_y).to(x.device) # 随机噪声
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt() # 扩散过程的标准差，决定每一步噪声的幅度
		sample = mean + sigma_t * z * 0.00001
		return (sample)


	def p_sample_loop(self, x, mask, shape):
		self.model.eval()
		prediction_total = torch.Tensor().cuda()
		for _ in range(20):
			cur_y = torch.randn(shape).to(x.device)
			for i in reversed(range(self.n_steps)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total
	
	def p_sample_loop_mean(self, x, mask, loc):
		prediction_total = torch.Tensor().cuda()
		for loc_i in range(1):
			cur_y = loc
			for i in reversed(range(NUM_Tau)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total

	def p_sample_loop_accelerate(self, x, mask, loc):
		'''
		Batch operation to accelerate the denoising process.

		x: [11, 10, 6]
		mask: [11, 11]
		cur_y: [11, 10, 20, 2]
		'''
		prediction_total = torch.Tensor().cuda()
		cur_y = loc[:, :10]
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
		cur_y_ = loc[:, 10:]
		for i in reversed(range(NUM_Tau)):
			cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
		# shape: B=b*n, K=10, T, 2
		prediction_total = torch.cat((cur_y_, cur_y), dim=1)
		return prediction_total



	def fit(self):
		# Training loop
		for epoch in range(0, self.cfg.num_epochs):
			loss= self._train_single_epoch(epoch)
			# print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
			# 	time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
			# 	epoch, loss_total, loss_distance, loss_uncertainty), self.log)
			print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\t'.format(
				time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
				epoch, loss), self.log)
			
			if (epoch + 1) % self.cfg.test_interval == 0:
				performance, samples = self._test_single_epoch()
				for time_i in range(4):
					print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
						time_i+1, performance['ADE'][time_i]/samples,
						time_i+1, performance['FDE'][time_i]/samples), self.log)
				cp_path = self.cfg.model_path % (epoch + 1)
				model_cp = {'model_diffusion_dict': self.model.state_dict()}
				torch.save(model_cp, cp_path)
			self.scheduler_model.step()


	def data_preprocess(self, data):
		"""
			pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
			fut_motion_3D: torch.Size([32, 11, 20, 2])
			fut_motion_mask: torch.Size([32, 11, 20])
			pre_motion_mask: torch.Size([32, 11, 10])
			traj_scale: 1
			pred_mask: None
			seq: nba
		"""
		batch_size = data['pre_motion_3D'].shape[0]

		traj_mask = torch.zeros(batch_size*11, batch_size*11).cuda()
		for i in range(batch_size):
			traj_mask[i*11:(i+1)*11, i*11:(i+1)*11] = 1.

		initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:]
		# augment input: absolute position, relative position, velocity
		past_traj_abs = ((data['pre_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, 10, 2)
		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 10, 2)
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

		fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 20, 2)
		return batch_size, traj_mask, past_traj, fut_traj
	
	def _train_single_epoch(self, epoch):
		
		self.model.train()
		loss, count = 0, 0
		
		# for data in self.train_loader:
		for batch_idx, (data) in enumerate(self.train_loader):
			batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
			loss = self.noise_estimation_loss(past_traj, fut_traj, traj_mask)

			# 记录每个batch的损失到TensorBoard
			writer_loss.add_scalar('Loss/denoising', loss, epoch * len(self.train_loader) + batch_idx)

			self.opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
			self.opt.step()
			count += 1
			if self.cfg.debug and count == 2:
				break

		return loss/count


	def _test_single_epoch(self):
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		count = 0
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
				
				loss = self.noise_estimation_loss(past_traj, fut_traj, traj_mask)
			
				pred_traj = self.p_sample_loop(past_traj, traj_mask, fut_traj.shape)

				fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
				# if count==100:
				# 	break
		return performance, samples


	def save_data(self):
		'''
		Save the visualization data.
		'''
		model_path = './results/checkpoints/led_vis.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		root_path = './visualization/data/'
				
		with torch.no_grad():
			for data in self.test_loader:
				_, traj_mask, past_traj, _, past_intention, past_similarity = self.data_preprocess(data)
				
				past_traj_movement = past_traj[..., :6]

				sample_prediction, mean_estimation, variance_estimation, intention_estimation, similarity_estimation = self.model_initializer(past_traj, past_intention, past_similarity, traj_mask)
				torch.save(sample_prediction, root_path+'p_var.pt')
				torch.save(mean_estimation, root_path+'p_mean.pt')
				torch.save(variance_estimation, root_path+'p_sigma.pt')
				torch.save(intention_estimation, root_path+'p_intention.pt')

				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				pred_traj = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, loc)
				pred_mean = self.p_sample_loop_mean(past_traj_movement, traj_mask, mean_estimation)

				torch.save(data['pre_motion_3D'], root_path+'past.pt')
				torch.save(data['fut_motion_3D'], root_path+'future.pt')
				torch.save(pred_traj, root_path+'prediction.pt')
				torch.save(pred_mean, root_path+'p_mean_denoise.pt')

				raise ValueError



	def test_single_model(self):
		model_path = './results/checkpoints/led_new.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		print_log(model_path, log=self.log)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		count = 0
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj, past_intention, past_similarity = self.data_preprocess(data)
				fut_traj_xy = fut_traj[..., :2]
				past_traj_movement = past_traj[..., :6]

				sample_prediction, mean_estimation, variance_estimation, intention_estimation, similarity_estimation = self.model_initializer(past_traj, past_intention, past_similarity, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
				pred_traj = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, loc)

				fut_traj_xy = fut_traj_xy.unsqueeze(1).repeat(1, 20, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj_xy - pred_traj, dim=-1) * self.traj_scale
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
					# if count==2:
					# 	break
		for time_i in range(4):
			print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples), log=self.log)
		
	