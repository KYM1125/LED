import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder, similarity_encoder, similarity_social_transformer, intention_social_transformer, intention_encoder, stepwise_encoder
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
		self.intention_dim = t_f * 3 * k_pred
		self.angel_dim = t_f * 1 * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer(t_h)
		self.intention_social_encoder = intention_social_transformer(t_h)
		self.similarity_social_encoder = similarity_social_transformer(t_h)
		self.ego_var_encoder = stepwise_encoder(d_h, t_h, t_f)
		self.ego_mean_encoder = stepwise_encoder(d_h, t_h, t_f)
		self.ego_scale_encoder = stepwise_encoder(d_h, t_h, t_f)
		self.ego_intention_encoder = stepwise_encoder(9, t_h, t_f)
		self.ego_similarity_encoder = stepwise_encoder(3, t_h, t_f)

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())
		self.intention_decoder = MLP(256*2, t_f * 3, hid_feat=(256, 128), activation=nn.ReLU())
		self.similarity_decoder = MLP(256*2, t_f * 1, hid_feat=(256, 128), activation=nn.ReLU())

	
	def forward(self, x, intention, past_similarity, mask=None):
		'''
			x: batch size, t_p, 6
			intention: batch size, t_p, 3
			past_similarity: batch size, t_p, 3
		'''

		x = x[:, :, :6]
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		intention_social_embed = self.intention_social_encoder(intention, mask)
		intention_social_embed = intention_social_embed.squeeze(1)
		similarity_social_embed = self.similarity_social_encoder(past_similarity, mask)
		similarity_social_embed = similarity_social_embed.squeeze(1)
		# B, 256
		
		ego_var_embed = self.ego_var_encoder(x)
		ego_mean_embed = self.ego_mean_encoder(x)
		ego_scale_embed = self.ego_scale_encoder(x)
		ego_intention_embed = self.ego_intention_encoder(intention)
		ego_similarity_embed = self.ego_similarity_encoder(past_similarity)
		
		# B, 256
		intention_total = torch.cat((ego_intention_embed, intention_social_embed), dim=-1)
		guess_intention = self.intention_decoder(intention_total).contiguous().view(-1, self.fut_len, 3) # B, T, 3
		
		similarity_total = torch.cat((ego_similarity_embed, similarity_social_embed), dim=-1)
		guess_similarity = self.similarity_decoder(similarity_total).contiguous().view(-1, self.fut_len, 1) # B, T, 1
		
		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2) # B, T, 2
		guess_mean_3d = torch.cat((guess_mean, torch.zeros_like(guess_mean[:, :, :1])), dim=-1)
		goal_sample = self.calculate_future_vectors_recursive(guess_mean_3d, guess_intention, guess_similarity)
		
		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total) # B, 1

		guess_scale_feat = self.scale_encoder(guess_scale) 
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1) 
		guess_var = self.var_decoder(var_total).contiguous().view(x.size(0), self.n, self.fut_len, 2) # B, K, T, 2

		return guess_var, guess_mean, guess_scale, guess_intention, guess_similarity, goal_sample[..., :2]
	
	def calculate_future_vectors_recursive(self, x, intention, similarity):
		"""
		根据递推方式计算未来的 T 帧 b 向量。通过逐步计算每一帧的 b 向量，使用上一帧的向量作为下一帧的输入。
		
		参数:
		x (torch.Tensor): 观测轨迹，形状为 (B, T, 3)，B是batch size，M是多模态数，T是预测帧数，3表示坐标维度
		intention (torch.Tensor): 意图向量，形状为 (B, T, 3)，M是多模态数，T是预测帧数，3表示坐标维度
		similarity (torch.Tensor): 相似度向量，形状为 (B, T, 1)，M是多模态数，T是预测帧数，1表示标量
		
		返回:
		torch.Tensor: 预测向量 b，形状为 (B, T, 3)
		"""
		B, T, _ = x.shape  # 获取批次大小、多模态数和预测帧数
		future_vectors = []  # 存储每一帧的b向量
		
		# 初始化第一个a向量为x的第0帧
		a = x[:, 0, :]  # 获取x的第0帧，形状 (B, 3)
		
		for t in range(1, T):
			# 获取当前的意图向量和相似度向量
			c = intention[:, t, :]  # 意图向量，形状 (B, 3)
			d = similarity[:, t, :]  # 相似度标量，形状 (B, 1)
			
			# 计算当前帧的b向量
			b = self.solve_b_batch(a, c, d)  # 计算b向量，调用之前的solve_b_batch函数
			
			# 将计算出的b向量添加到未来轨迹列表
			future_vectors.append(b.unsqueeze(1))  # 添加到未来轨迹列表
			
			# 更新a向量为当前计算出的b向量，作为下一个时间步的输入
			a =  x[:, t, :]  # 更新a向量为当前帧的x向量
		
		# 将未来向量拼接成 (B, T-1, 3)
		future_vectors = torch.cat(future_vectors, dim=1)  # 形状为 (B, T-1, 3)
		
		# 插入x的第0帧到未来轨迹的最前面
		future_vectors = torch.cat([x[:, 0, :].unsqueeze(1), future_vectors], dim=1)  # 形状为 (B, T, 3)
		
		return future_vectors


	def solve_b_batch(self, a, c, d):
		"""
		批量计算 b 向量，优化的版本使用矩阵运算避免逐个样本的计算。
		
		参数:
		a (torch.Tensor): 向量 a，形状为 (B, 3)
		c (torch.Tensor): 向量 a 叉乘 b 的结果 c，形状为 (B, 3)
		d (torch.Tensor): 向量 a 点乘 b 的结果 d，形状为 (B, 1)
		
		返回:
		torch.Tensor: 向量 b，形状为 (B, 3)
		"""
		# 计算向量 a 和 c 的模长
		a_norm = torch.norm(a, dim=-1, keepdim=True)  # (B, 1)
		c_norm = torch.norm(c, dim=-1, keepdim=True)  # (B, 1)

		# 计算 b 的模长
		b_norm = torch.sqrt(c_norm**2 + d**2) / a_norm  # (B, 1)

		# 计算叉积方向
		b_direction = torch.cross(a, c, dim=-1)  # (B, 3)
		b_direction_norm = torch.norm(b_direction, dim=-1, keepdim=True)  # (B, 1)
		b_direction = b_direction / b_direction_norm  # 归一化方向，(B, 3)

		# 批量计算 b 向量
		b = b_norm * b_direction  # (B, 3)

		return b
	
	def rotate_vectors(self, guess_intention, guess_angel, guess_x_3d, guess_y_3d):
		# 创建旋转四元数
		device = guess_intention.device
		normalized_guess_intention = guess_intention / guess_intention.norm(dim=-1, keepdim=True)
		
		# 展平旋转向量为(N, 3)形状
		rotvec = (normalized_guess_intention.detach().cpu().numpy() * guess_angel.detach().cpu().numpy()).reshape(-1, 3)

		# 使用 `from_rotvec` 计算旋转
		r = R.from_rotvec(rotvec)

		# 应用旋转，得到旋转后的向量
		rotated_x = r.apply(guess_x_3d.detach().cpu().numpy().reshape(-1, 3))  # 示例：旋转guess_mean_x_3d
		rotated_y = r.apply(guess_y_3d.detach().cpu().numpy().reshape(-1, 3))  # 旋转guess_mean_y_3d

		# 将旋转结果调整回原始形状
		rotated_x = rotated_x.reshape(guess_x_3d.shape)
		rotated_y = rotated_y.reshape(guess_y_3d.shape)

		# 将旋转后的向量转换回torch张量，并确保它们在正确的设备上
		rotated_x = torch.tensor(rotated_x, device=device)
		rotated_y = torch.tensor(rotated_y, device=device)

		return rotated_x, rotated_y

	



