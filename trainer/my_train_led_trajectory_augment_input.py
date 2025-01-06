
import os
import time
# import tqdm
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from utils.config import Config
from utils.utils import print_log


from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate
import matplotlib.pyplot as plt
from IPython.display import Image, display
from torch.utils.tensorboard import SummaryWriter

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
# from models.model_led_initializer_with_intention import VecIntInitializer as MyInitializationModel
from models.model_led_initializer_new_stepwise import VecIntInitializer as MyInitializationModel
from models.model_led_stepweise import StepweiseInitializer

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

		self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
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
	
	def plot_angle_comparison_to_tensorboard(self, angel_estimation, target_angel, step, title='Angle Comparison'):
		"""
		绘制 angle_estimation 和 target_angel 的变化曲线，并将图像保存到 TensorBoard 中。

		:param angel_estimation: 模型预测的角度 (B, T)
		:param target_angel: 真实角度 (B, T)
		:param step: 当前训练步骤，作为 TensorBoard 的全局步骤
		:param title: 图表标题
		"""
		# 将角度从弧度转换为度数
		angel_estimation_deg = torch.rad2deg(angel_estimation.detach()).cpu().numpy()  # 使用 detach()
		target_angel_deg = torch.rad2deg(target_angel.detach()).cpu().numpy()  # 使用 detach()

		# 设置画布
		plt.figure(figsize=(10, 6))

		# 绘制预测角度和真实角度
		plt.plot(angel_estimation_deg[0], label='Predicted Angle', color='blue', linestyle='--')
		plt.plot(target_angel_deg[0], label='True Angle', color='red', linestyle='-')

		# 添加图例和标题
		plt.legend()
		plt.title(title)
		plt.xlabel('Time Step')
		plt.ylabel('Angle (degrees)')
		plt.grid(True)

		# 将图像写入 TensorBoard
		writer.add_figure('Angle Comparison', plt.gcf(), global_step=step)

		# 清理当前图形，以防影响后续绘图
		plt.close()

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
		batch_size = x.shape[0]
		# Select a random step for each example
		t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
		t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
		# x0 multiplier
		a = self.extract(self.alphas_bar_sqrt, t, y_0)
		beta = self.extract(self.betas, t, y_0)
		# eps multiplier
		am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
		e = torch.randn_like(y_0)
		# model input
		y = y_0 * a + e * am1
		output = self.model(y, beta, x, mask)
		# batch_size, 20, 2
		return (e - output).square().mean()



	def p_sample(self, x, mask, cur_y, t):
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
	
	def p_sample_accelerate(self, x, mask, cur_y, t): # 去噪过程
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



	# def fit(self):
	# 	# Training loop
	# 	for epoch in range(0, self.cfg.num_epochs):
	# 		loss_total, loss_distance, loss_uncertainty, loss_intention, loss_simility, loss_goal_distance, loss_goal_uncertainty = self._train_single_epoch(epoch)
	# 		# print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
	# 		# 	time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
	# 		# 	epoch, loss_total, loss_distance, loss_uncertainty), self.log)
	# 		print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}\tLoss Intention: {:.6f}\tLoss Simility: {:.6f}\tLoss Goal Distance: {:.6f}\tLoss Goal Uncertainty: {:.6f}'.format(
	# 			time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
	# 			epoch, loss_total, loss_distance, loss_uncertainty, loss_intention, loss_simility, loss_goal_distance, loss_goal_uncertainty), self.log)
			
	# 		if (epoch + 1) % self.cfg.test_interval == 0:
	# 			performance, samples, goal_samples = self._test_single_epoch()
	# 			for time_i in range(4):
	# 				print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}\t--Goal ADE({}s): {:.4f}\t--Goal FDE({}s): {:.4f}'.format(
	# 							time_i+1, performance['ADE'][time_i]/samples,
	# 							time_i+1, performance['FDE'][time_i]/samples,
	# 							time_i+1, performance['goal_ADE'][time_i]/goal_samples,
	# 							time_i+1, performance['goal_FDE'][time_i]/goal_samples), self.log)
	# 			cp_path = self.cfg.model_path % (epoch + 1)
	# 			model_cp = {'model_initializer_dict': self.model_initializer.state_dict()}
	# 			torch.save(model_cp, cp_path)
	# 		self.scheduler_model.step()
	def fit(self):
		 # Training loop
		with tqdm(total=self.cfg.num_epochs, desc="Training Progress") as pbar:
			for epoch in range(self.cfg.num_epochs):
				# Train a single epoch
				(
					loss_total, loss_distance, loss_uncertainty, loss_intention,
					loss_simility, loss_goal_distance, loss_goal_uncertainty
				) = self._train_single_epoch(epoch)

				# Update progress bar description
				pbar.set_postfix({
					'Loss': f"{loss_total:.6f}",
					'Dist Loss': f"{loss_distance:.6f}",
					'Uncert Loss': f"{loss_uncertainty:.6f}",
					'Intent Loss': f"{loss_intention:.6f}",
					'Simil Loss': f"{loss_simility:.6f}",
					'Goal Dist': f"{loss_goal_distance:.6f}",
					'Goal Uncert': f"{loss_goal_uncertainty:.6f}"
				})
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}\tLoss Intention: {:.6f}\tLoss Simility: {:.6f}\tLoss Goal Distance: {:.6f}\tLoss Goal Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch, loss_total, loss_distance, loss_uncertainty, loss_intention, loss_simility, loss_goal_distance, loss_goal_uncertainty), self.log, display=False)
				pbar.set_description(f"Training Epoch {epoch + 1}/{self.cfg.num_epochs}")
				pbar.update(1)

				if (epoch + 1) % self.cfg.test_interval == 0:
					performance, samples, goal_samples = self._test_single_epoch()
					for time_i in range(4):
						ade = performance['ADE'][time_i] / samples
						fde = performance['FDE'][time_i] / samples
						goal_ade = performance['goal_ADE'][time_i] / goal_samples
						goal_fde = performance['goal_FDE'][time_i] / goal_samples

						ade_reduction = ((ade - goal_ade) / ade) * 100  # Correct formula
						fde_reduction = ((fde - goal_fde) / fde) * 100

						# Log test results
						print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}\t--Goal ADE({}s): {:.4f} (下降 {:.2f}%)\t--Goal FDE({}s): {:.4f} (下降 {:.2f}%)'.format(
							time_i + 1, ade,
							time_i + 1, fde,
							time_i + 1, goal_ade, ade_reduction,
							time_i + 1, goal_fde, fde_reduction), self.log)

					# Save the model checkpoint
					cp_path = self.cfg.model_path % (epoch + 1)
					model_cp = {'model_initializer_dict': self.model_initializer.state_dict()}
					torch.save(model_cp, cp_path)

				# Update learning rate scheduler
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
		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 10, 2) # initial_pos是观测帧与预测帧的交叉点，是观测帧的最后一帧
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
		past_traj_acc = torch.cat((past_traj_vel[:, 1:] - past_traj_vel[:, :-1], torch.zeros_like(past_traj_vel[:, -1:])), dim=1) # 后一帧减去前一帧，得到加速度
		
		past_traj_pos_angle = torch.atan2(past_traj_rel[..., 1], past_traj_rel[..., 0])
		past_traj_vel_angle = torch.atan2(past_traj_vel[..., 1], past_traj_vel[..., 0])
		past_traj_acc_angle = torch.atan2(past_traj_acc[..., 1], past_traj_acc[..., 0]) # 相对于x轴的逆时针夹角，单位为弧度。
		
		position_angle_change = torch.cat((past_traj_pos_angle[:, 1:] - past_traj_pos_angle[:, :-1], torch.zeros_like(past_traj_pos_angle[:, -1:])), dim=1).unsqueeze(-1)
		velocity_angle_change = torch.cat((past_traj_vel_angle[:, 1:] - past_traj_vel_angle[:, :-1], torch.zeros_like(past_traj_vel_angle[:, -1:])), dim=1).unsqueeze(-1)
		acceleration_angle_change = torch.cat((past_traj_acc_angle[:, 1:] - past_traj_acc_angle[:, :-1], torch.zeros_like(past_traj_acc_angle[:, -1:])), dim=1).unsqueeze(-1)
		
		past_traj_abs_3d = torch.cat((past_traj_abs, torch.zeros_like(past_traj_abs[..., :1])), dim=-1)
		past_traj_rel_3d = torch.cat((past_traj_rel, torch.zeros_like(past_traj_rel[..., :1])), dim=-1)
		past_traj_vel_3d = torch.cat((past_traj_vel, torch.zeros_like(past_traj_vel[..., :1])), dim=-1)

		past_traj_similarity_abs = torch.sum(past_traj_abs_3d[:, :-1] * past_traj_abs_3d[:, 1:], dim=-1)
		past_traj_similarity_rel = torch.sum(past_traj_rel_3d[:, :-1] * past_traj_rel_3d[:, 1:], dim=-1)
		past_traj_similarity_vel = torch.sum(past_traj_vel_3d[:, :-1] * past_traj_vel_3d[:, 1:], dim=-1)
		past_traj_similarity_abs = torch.cat((past_traj_similarity_abs, torch.zeros_like(past_traj_similarity_abs[:, -1:])), dim=1).unsqueeze(-1)
		past_traj_similarity_rel = torch.cat((past_traj_similarity_rel, torch.zeros_like(past_traj_similarity_rel[:, -1:])), dim=1).unsqueeze(-1)
		past_traj_similarity_vel = torch.cat((past_traj_similarity_vel, torch.zeros_like(past_traj_similarity_vel[:, -1:])), dim=1).unsqueeze(-1)

		
		past_traj_intention_abs = torch.cross(past_traj_abs_3d[:, :-1], past_traj_abs_3d[:, 1:], dim=-1) # 当前帧叉乘下一帧
		past_traj_intention = torch.cross(past_traj_rel_3d[:, :-1], past_traj_rel_3d[:, 1:], dim=-1) 
		past_traj_intention_vel = torch.cross(past_traj_vel_3d[:, :-1], past_traj_vel_3d[:, 1:], dim=-1)
		past_traj_intention_abs = torch.cat((past_traj_intention_abs, torch.zeros_like(past_traj_intention_abs[:, -1:, :])), dim=1)
		past_traj_intention = torch.cat((past_traj_intention, torch.zeros_like(past_traj_intention[:, -1:, :])), dim=1)
		past_traj_intention_vel = torch.cat((past_traj_intention_vel, torch.zeros_like(past_traj_intention_vel[:, -1:, :])), dim=1)

		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
		past_similarity = torch.cat((past_traj_similarity_abs, past_traj_similarity_rel, past_traj_similarity_vel), dim=-1)
		past_intention = torch.cat((past_traj_intention_abs, past_traj_intention, past_traj_intention_vel), dim=-1)

		fut_traj_abs = ((data['fut_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, 20, 2)
		fut_traj_rel = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 20, 2)
		fut_traj_vel = torch.cat((fut_traj_rel[:, 1:] - fut_traj_rel[:, :-1], torch.zeros_like(fut_traj_rel[:, -1:])), dim=1)
		fut_traj_acc = torch.cat((fut_traj_vel[:, 1:] - fut_traj_vel[:, :-1], torch.zeros_like(fut_traj_vel[:, -1:])), dim=1)
		
		fut_traj_pos_angle = torch.atan2(fut_traj_rel[..., 1], fut_traj_rel[..., 0])
		fut_traj_vel_angle = torch.atan2(fut_traj_vel[..., 1], fut_traj_vel[..., 0])
		fut_traj_acc_angle = torch.atan2(fut_traj_acc[..., 1], fut_traj_acc[..., 0])
		
		position_angle_change_fut = torch.cat((fut_traj_pos_angle[:, 1:] - fut_traj_pos_angle[:, :-1], torch.zeros_like(fut_traj_pos_angle[:, -1:])), dim=1).unsqueeze(-1)
		velocity_angle_change_fut = torch.cat((fut_traj_vel_angle[:, 1:] - fut_traj_vel_angle[:, :-1], torch.zeros_like(fut_traj_vel_angle[:, -1:])), dim=1).unsqueeze(-1)
		acceleration_angle_change_fut = torch.cat((fut_traj_acc_angle[:, 1:] - fut_traj_acc_angle[:, :-1], torch.zeros_like(fut_traj_acc_angle[:, -1:])), dim=1).unsqueeze(-1)
		
		fut_traj_rel_3d = torch.cat((fut_traj_rel, torch.zeros_like(fut_traj_rel[..., :1])), dim=-1)
		fut_similarity = torch.sum(fut_traj_rel_3d[:, :-1] * fut_traj_rel_3d[:, 1:], dim=-1).unsqueeze(-1) # 当前帧点乘下一帧
		fut_similarity = torch.cat((fut_similarity, torch.zeros_like(fut_similarity[:, -1:, :])), dim=1)
		fut_intention = torch.cross(fut_traj_rel_3d[:, :-1], fut_traj_rel_3d[:, 1:], dim=-1) # 当前帧叉乘下一帧
		fut_intention = torch.cat((fut_intention, torch.zeros_like(fut_intention[:, -1:, :])), dim=1)
		
		fut_traj = torch.cat((fut_traj_rel, fut_similarity, fut_intention), dim=-1)
		return batch_size, traj_mask, past_traj, fut_traj, past_intention, past_similarity,

	def compute_intention_loss(self, intention_estimation, target_intention, L):
		"""
		计算行人意图的损失，使得预测的意图向量与目标意图向量尽可能接近。
		
		:param intention_estimation: 预测的意图估计 (B, M, T, 3)
		:param target_intention: 目标意图向量 (B, T, 3)
		:return: intention loss
		"""
		# 计算每个模态的意图向量与目标意图向量之间的欧几里得距离 (B, M, T)
		# 注意，target_intention 需要扩展为 (B, M, T, 3) 以便与 intention_estimation 进行比较
		target_intention_expanded = target_intention.unsqueeze(1)  # (B, 1, T, 3)
		target_intention_expanded = target_intention_expanded.expand_as(intention_estimation)  # (B, M, T, 3)
		
		# 计算每个时间步的意图向量的欧几里得距离
		distance = torch.norm(intention_estimation[:,:,:-1,:] - target_intention_expanded[:,:,:-1,:], p=L, dim=-1)  # (B, M, T)
		
		# 计算每个时间步的损失：距离越小，损失越小
		intention_loss = distance.mean(dim=1)  # 对每个模态取平均，得到 (B, T)

		# 使用时间加权的损失
		weighted_intention_loss = (intention_loss * self.temporal_reweight[:,:,:-1].squeeze(0)).mean()

		return weighted_intention_loss

	
	def compute_angle_cosine_loss(self, angel_estimation, target_angel):
		"""
		计算预测的角度与目标角度之间的余弦相似度损失。
		:param angel_estimation: 模型预测的角度 (B, M, T)
		:param target_angel: 真实角度 (B, T)
		:return: 余弦相似度损失
		"""
		# 扩展 target_angel 的维度为 (B, M, T)，与 angel_estimation 的形状一致
		target_angel_expanded = target_angel.unsqueeze(1)  # (B, 1, T)
		target_angel_expanded = target_angel_expanded.expand_as(angel_estimation)  # (B, M, T)
		
		# 计算角度的余弦相似度
		cos_sim = torch.cos(angel_estimation - target_angel_expanded)
		
		# 余弦相似度越高，说明预测和目标越接近，反之亦然
		cosine_loss = 1 - cos_sim  # 使用1减去余弦相似度来表示损失
		
		# 对所有模态和时间步取平均
		cosine_loss = cosine_loss.mean(dim=1) * self.temporal_reweight.squeeze(0)  # 计算总的平均损失
		
		return cosine_loss.mean()

	
	def _train_single_epoch(self, epoch):
		
		self.model.train()
		self.model_initializer.train()
		loss_total, loss_dt, loss_dc, loss_it, loss_sm, loss_goal_dt, loss_goal_dc, count = 0, 0, 0, 0, 0, 0, 0, 0
		
		# for data in self.train_loader:
		for batch_idx, (data) in enumerate(self.train_loader):
			batch_size, traj_mask, past_traj, fut_traj, past_intention, past_similarity = self.data_preprocess(data)
			fut_traj_xy = fut_traj[..., :2]
			target_similarity = fut_traj[..., 2].unsqueeze(-1)
			target_intention = fut_traj[..., 3:6]
			past_traj_movement = past_traj[..., :6]

			sample_prediction, mean_estimation, variance_estimation, intention_estimation, similarity_estimation, goal_estimation = self.model_initializer(past_traj, past_intention, past_similarity, traj_mask)
			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			loc = sample_prediction + mean_estimation[:, None]
			generated_y = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, loc)

			goal_prediction = torch.exp(variance_estimation/2)[..., None, None] * goal_estimation / goal_estimation.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			goal_loc = goal_prediction + mean_estimation[:, None]
			generated_goal = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, goal_loc)
			
			loss_goal_dist = (	(generated_goal - fut_traj_xy.unsqueeze(dim=1)).norm(p=2, dim=-1)
								* 
							 self.temporal_reweight
						).mean(dim=-1).min(dim=1)[0].mean()
			loss_goal_uncertainty = (torch.exp(-variance_estimation)
		       						*
								(generated_goal - fut_traj_xy.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2)) 
									+ 
								variance_estimation
								).mean()
			# test = (generated_y - fut_traj_xy.unsqueeze(dim=1)).norm(p=2, dim=-1) 
			loss_dist = (	(generated_y - fut_traj_xy.unsqueeze(dim=1)).norm(p=2, dim=-1) 
								* 
							 self.temporal_reweight
						).mean(dim=-1).min(dim=1)[0].mean()
			loss_uncertainty = (torch.exp(-variance_estimation)
		       						*
								(generated_y - fut_traj_xy.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2)) 
									+ 
								variance_estimation
								).mean()
			loss_intention = self.compute_intention_loss(intention_estimation, target_intention, 2) 
			# loss_angel = self.compute_angle_cosine_loss(angel_estimation.squeeze(-1), target_angel)
			# # 平均化M维度
			# similarity_estimation_avg = similarity_estimation.mean(dim=1)  # (B, T, 1)

			# # 计算L1损失
			# loss_similarity = ( 
			# 	torch.abs(similarity_estimation_avg[:,:-1,:] - target_similarity[:,:-1,:]) * self.temporal_reweight[:,:,:-1].squeeze(0).unsqueeze(-1)
			# ).mean(dim=-1).min(dim=1)[0].mean()
			loss_similarity = self.compute_intention_loss(similarity_estimation, target_similarity, 1)


			loss = loss_dist*50 + loss_uncertainty + loss_intention*10 + loss_similarity*10 + loss_goal_dist*50 + loss_goal_uncertainty
			loss_total += loss.item()
			loss_dt += loss_dist.item()*50
			loss_dc += loss_uncertainty.item()
			loss_it += loss_intention.item()*10
			loss_sm += loss_similarity.item()*10
			loss_goal_dt += loss_goal_dist.item()*50
			loss_goal_dc += loss_goal_uncertainty.item()

			# 记录每个batch的损失到TensorBoard
			writer_loss.add_scalar('Loss/train_total', loss_total, epoch * len(self.train_loader) + batch_idx)
			writer_loss.add_scalar('Loss/train_dist', loss_dt, epoch * len(self.train_loader) + batch_idx)
			writer_loss.add_scalar('Loss/train_uncertainty', loss_dc, epoch * len(self.train_loader) + batch_idx)
			writer_loss.add_scalar('Loss/train_intention', loss_it, epoch * len(self.train_loader) + batch_idx)
			writer_loss.add_scalar('Loss/train_simularity', loss_sm, epoch * len(self.train_loader) + batch_idx)
			writer_loss.add_scalar('Loss/train_goal_dist', loss_goal_dt, epoch * len(self.train_loader) + batch_idx)
			writer_loss.add_scalar('Loss/train_goal_uncertainty', loss_goal_dc, epoch * len(self.train_loader) + batch_idx)

			self.opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
			self.opt.step()
			count += 1
			if self.cfg.debug and count == 20:
				break

		return loss_total/count, loss_dt/count, loss_dc/count, loss_it/count, loss_sm/count, loss_goal_dt/count, loss_goal_dc/count


	def _test_single_epoch(self):
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0],
						'goal_FDE': [0, 0, 0, 0],
						'goal_ADE': [0, 0, 0, 0]}
		samples = 0
		goal_samples = 0
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
				target_angel = fut_traj[..., 2]
				past_traj_movement = past_traj[..., :6]

				sample_prediction, mean_estimation, variance_estimation, intention_estimation, similarity_estimation, goal_estimation = self.model_initializer(past_traj, past_intention, past_similarity, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
				pred_traj = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, loc)

				goal_prediction = torch.exp(variance_estimation/2)[..., None, None] * goal_estimation / goal_estimation.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				goal_loc = goal_prediction + mean_estimation[:, None]
				generated_goal = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, goal_loc)

				fut_traj_xy = fut_traj_xy.unsqueeze(1).repeat(1, 20, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj_xy - pred_traj, dim=-1) * self.traj_scale
				
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]

				goal_distances = torch.norm(fut_traj_xy - generated_goal, dim=-1) * self.traj_scale

				for time_i in range(1, 5):
					ade = (goal_distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (goal_distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['goal_ADE'][time_i-1] += ade.item()
					performance['goal_FDE'][time_i-1] += fde.item()
				goal_samples += goal_distances.shape[0]
				count += 1
				# if count==100:
				# 	break
		return performance, samples, goal_samples


	def save_data(self):
		'''
		Save the visualization data.
		'''
		model_path = './results/my_led_augment_debug/12_24_new_prototype_with_goal/models/model_0100.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		root_path = './visualization/my_data/'
				
		with torch.no_grad():
			for data in self.test_loader:
				_, traj_mask, past_traj, _, past_intention, past_similarity = self.data_preprocess(data)
				
				past_traj_movement = past_traj[..., :6]

				sample_prediction, mean_estimation, variance_estimation, intention_estimation, similarity_estimation, goal_estimation = self.model_initializer(past_traj, past_intention, past_similarity, traj_mask)
				torch.save(sample_prediction, root_path+'p_var.pt')
				torch.save(mean_estimation, root_path+'p_mean.pt')
				torch.save(variance_estimation, root_path+'p_sigma.pt')
				torch.save(intention_estimation, root_path+'p_intention.pt')
				torch.save(similarity_estimation, root_path+'p_similarity.pt')
				torch.save(goal_estimation, root_path+'p_goal.pt')
				
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
				
				goal_prediction = torch.exp(variance_estimation/2)[..., None, None] * goal_estimation / goal_estimation.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				goal_loc = goal_prediction + mean_estimation[:, None]
				pred_goal = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, goal_loc)

				pred_traj = self.p_sample_loop_accelerate(past_traj_movement, traj_mask, loc)
				pred_mean = self.p_sample_loop_mean(past_traj_movement, traj_mask, mean_estimation)

				torch.save(data['pre_motion_3D'], root_path+'past.pt')
				torch.save(data['fut_motion_3D'], root_path+'future.pt')
				torch.save(pred_traj, root_path+'prediction.pt')
				torch.save(pred_goal, root_path+'goal.pt')
				torch.save(pred_mean, root_path+'p_mean_denoise.pt')

				# 临时终止程序以调试生成数据
				print("Debug: Data saving completed for one batch. Stopping execution for inspection.")
				exit()




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
		
	