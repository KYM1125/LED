import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder, angel_encoder
from scipy.spatial.transform import Rotation as R

class VecIntInitializer(nn.Module):
	def __init__(self, t_h: int=8, d_h: int=6, t_f: int=40, d_f: int=2, k_pred: int=20):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions.

		'''
		super(VecIntInitializer, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer(t_h)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()
		self.ego_intention_encoder = st_encoder()
		self.ego_angel_encoder = angel_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())
		self.intention_decoder = MLP(256*2, t_f * 3, hid_feat=(256, 128), activation=nn.ReLU())
		self.angel_decoder = MLP(256*2, t_f * 1, hid_feat=(256, 128), activation=nn.ReLU())

	
	def forward(self, x, mask=None):
		'''
		x: batch size, t_p, 6
		'''
		# relative_features = self.calculate_relative_changes(x)
		# x_x = x[:, :, 0]
		# v_x = x[:, :, 2]
		# a_x = x[:, :, 4]
		# t = (x_x[:, :-1] - x_x[:, 1:]) / (v_x[:, :-1] + 1e-6)
		# angles, direction = self.calculate_acceleration_angle_with_direction(x[:,:,4], x[:,:,5])
		# signed_angles = angles * direction
		# relative_angels, relative_direction = self.calculate_acceleration_angle_with_direction(relative_features[:,:,4], relative_features[:,:,5])
		signed_angles = x[:, :, 6].unsqueeze(-1)
		x = x[:, :, :6]
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		# B, 256
		
		ego_var_embed = self.ego_var_encoder(x)
		ego_mean_embed = self.ego_mean_encoder(x)
		ego_scale_embed = self.ego_scale_encoder(x)
		ego_intention_embed = self.ego_intention_encoder(x)
		ego_angel_embed = self.ego_angel_encoder(signed_angles)
		
		# B, 256
		intention_total = torch.cat((ego_intention_embed, social_embed), dim=-1)
		guess_intention = self.intention_decoder(intention_total).contiguous().view(-1, self.fut_len, 3) # B, T, 3
		
		angel_total = torch.cat((ego_angel_embed, social_embed), dim=-1)
		guess_angel = self.angel_decoder(angel_total).contiguous().view(-1, self.fut_len, 1) # B, T, 1

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2) # B, T, 2
		
		guess_mean_x_3d = torch.stack((guess_mean[:,:,0], torch.zeros_like(guess_mean[:,:,0]), torch.zeros_like(guess_mean[:,:,0])), dim=-1)  # Shape: [22, 20, 3]
		guess_mean_y_3d = torch.stack((torch.zeros_like(guess_mean[:,:,1]), guess_mean[:,:,1], torch.zeros_like(guess_mean[:,:,1])), dim=-1)  # Shape: [22, 20, 3]
		# rotated_x = torch.cross(guess_action, guess_mean_x_3d, dim=-1) # guess_intention应该乘以t的平方并除以2，x = x0 + v0*t + 1/2*a*t^2
		# rotated_y = torch.cross(guess_action, guess_mean_y_3d, dim=-1)
		# guess_mean_rotated = guess_mean.clone()
		# guess_mean_rotated[:, :, 0] = rotated_x[:, :, 1] 
		# guess_mean_rotated[:, :, 1] = rotated_y[:, :, 0]
		rotated_x, rotated_y = self.rotate_vectors(guess_intention, guess_angel, guess_mean_x_3d, guess_mean_y_3d)
		guess_mean_rotated = guess_mean.clone()
		guess_mean_rotated = (rotated_x + rotated_y)[:, :, :2]
		
		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total) # B, 1

		guess_scale_feat = self.scale_encoder(guess_scale) 
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1) 
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2) # B, K, T, 2

		return guess_var, guess_mean_rotated, guess_scale, guess_intention, guess_angel
	
	def calculate_relative_changes(self, x):
		"""
		计算每个时间步相对于上一时刻的变化量，x包含每个时间步的 [x, y, vx, vy, ax, ay]

		参数：
		x: Tensor, shape: (batch_size, time_steps, 6)，每个时间步的 [x, y, vx, vy, ax, ay]

		返回：
		relative_features: Tensor, shape: (batch_size, time_steps, 6)，每个时间步的相对变化量 [delta_x, delta_y, delta_vx, delta_vy, delta_ax, delta_ay]
		"""
		device = x.device
		# 获取每一时刻的位置、速度、加速度
		x_pos = x[:, :, 0]  # 位置 x
		y_pos = x[:, :, 1]  # 位置 y
		vx = x[:, :, 2]  # 速度 vx
		vy = x[:, :, 3]  # 速度 vy
		ax = x[:, :, 4]  # 加速度 ax
		ay = x[:, :, 5]  # 加速度 ay

		# 计算相对变化量
		delta_x = x_pos[:, 1:] - x_pos[:, :-1]  # 当前 x - 上一时刻 x
		delta_y = y_pos[:, 1:] - y_pos[:, :-1]  # 当前 y - 上一时刻 y
		delta_vx = vx[:, 1:] - vx[:, :-1]  # 当前 vx - 上一时刻 vx
		delta_vy = vy[:, 1:] - vy[:, :-1]  # 当前 vy - 上一时刻 vy
		delta_ax = ax[:, 1:] - ax[:, :-1]  # 当前 ax - 上一时刻 ax
		delta_ay = ay[:, 1:] - ay[:, :-1]  # 当前 ay - 上一时刻 ay

		# 对于 t=0 的位置、速度、加速度变化，假设变化量为0
		delta_x = torch.cat([torch.zeros(x.size(0), 1, device=device), delta_x], dim=1)
		delta_y = torch.cat([torch.zeros(x.size(0), 1, device=device), delta_y], dim=1)
		delta_vx = torch.cat([torch.zeros(x.size(0), 1, device=device), delta_vx], dim=1)
		delta_vy = torch.cat([torch.zeros(x.size(0), 1, device=device), delta_vy], dim=1)
		delta_ax = torch.cat([torch.zeros(x.size(0), 1, device=device), delta_ax], dim=1)
		delta_ay = torch.cat([torch.zeros(x.size(0), 1, device=device), delta_ay], dim=1)

		# 将相对变化量合并成一个新的张量
		relative_features = torch.stack([delta_x, delta_y, delta_vx, delta_vy, delta_ax, delta_ay], dim=-1)

		return relative_features
	
	def calculate_acceleration_angle_with_direction(self, a_x, a_y):
		"""
		计算加速度向量之间的转向角（弧度制），并判断旋转方向（顺时针/逆时针）
		
		参数：
		a_x: Tensor，形状 (batch_size, time_steps)，加速度向量在x轴上的分量
		a_y: Tensor，形状 (batch_size, time_steps)，加速度向量在y轴上的分量
		
		返回：
		angles: Tensor，形状 (batch_size, time_steps-1)，每个时间步之间加速度向量的转向角（弧度制）
		direction: Tensor，形状 (batch_size, time_steps-1)，每个时间步之间加速度向量的旋转方向（+1表示逆时针，-1表示顺时针）
		"""
		device = a_x.device
		# 计算加速度向量的模长
		a_mag = torch.sqrt(a_x ** 2 + a_y ** 2)  # (batch_size, time_steps)
		
		# 计算时间步之间的加速度向量的点积
		# 左闭右开区间，a_x[:, 1:]表示时间步 0 到 time_steps-2 的加速度向量在 x 方向的分量，a_x[:, :-1]表示时间步 1 到 time_steps-1 的加速度向量在 x 方向的分量，所以就是t时刻的向量在t+1时刻上向量的投影
		dot_product = a_x[:, 1:] * a_x[:, :-1] + a_y[:, 1:] * a_y[:, :-1]  # (batch_size, time_steps-1)
		
		# 计算加速度向量的模长乘积，t时刻的向量模长乘以t+1时刻的向量模长
		a_mag_prod = a_mag[:, 1:] * a_mag[:, :-1]  # (batch_size, time_steps-1)
		
		# 计算夹角的cos值，t时刻到t+1时刻的向量夹角的cos值
		cos_angle = dot_product / (a_mag_prod + 1e-6)  # 防止除零错误

		# 计算角度（弧度制）
		angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))  # 计算反余弦，返回弧度制角度
		angles[:, -1] = 0 # 第8帧到第9帧（最后一帧）的转向角没有意义，因为第9帧的加速度是认为填充的0
		
		# 计算旋转方向
		# 外积：a_t × a_t+1 = a_x(t) * a_y(t+1) - a_y(t) * a_x(t+1)
		cross_product = a_x[:, 1:] * a_y[:, :-1] - a_y[:, 1:] * a_x[:, :-1]  # (batch_size, time_steps-1) 第一项为0
		
		# 根据外积的符号来确定旋转方向
		direction = torch.sign(cross_product)  # +1 表示逆时针，-1 表示顺时针
		
		return angles, direction
	
	def rotate_vectors(self, guess_intention, guess_angel, guess_mean_x_3d, guess_mean_y_3d):
		# 创建旋转四元数
		device = guess_intention.device
		normalized_guess_intention = guess_intention / guess_intention.norm(dim=-1, keepdim=True)
		
		# 展平旋转向量为(N, 3)形状
		rotvec = (normalized_guess_intention.detach().cpu().numpy() * guess_angel.detach().cpu().numpy()).reshape(-1, 3)

		# 使用 `from_rotvec` 计算旋转
		r = R.from_rotvec(rotvec)

		# 应用旋转，得到旋转后的向量
		rotated_x = r.apply(guess_mean_x_3d.detach().cpu().numpy().reshape(-1, 3))  # 示例：旋转guess_mean_x_3d
		rotated_y = r.apply(guess_mean_y_3d.detach().cpu().numpy().reshape(-1, 3))  # 旋转guess_mean_y_3d

		# 将旋转结果调整回原始形状
		rotated_x = rotated_x.reshape(guess_mean_x_3d.shape)
		rotated_y = rotated_y.reshape(guess_mean_y_3d.shape)

		# 将旋转后的向量转换回torch张量，并确保它们在正确的设备上
		rotated_x = torch.tensor(rotated_x, device=device)
		rotated_y = torch.tensor(rotated_y, device=device)

		return rotated_x, rotated_y

	



