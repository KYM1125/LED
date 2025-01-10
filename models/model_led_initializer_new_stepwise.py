import torch
import torch.nn as nn
from models.layers import MLP, stepwise_encoder, stepweise_transformer, social_transformer, st_encoder, similarity_encoder, intention_encoder
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

		# self.old_social_encoder = social_transformer(t_h)
		self.social_encoder = social_transformer(t_h)
		self.intention_social_encoder = social_transformer(t_h)
		self.similarity_social_encoder = social_transformer(t_h)
		self.ego_var_encoder = stepwise_encoder(d_h, t_h, t_f)
		self.ego_mean_encoder = stepwise_encoder(d_h, t_h, t_f)
		self.ego_scale_encoder = stepwise_encoder(d_h, t_h, t_f)
		self.ego_intention_encoder = stepwise_encoder(9, t_h, t_f)
		self.ego_similarity_encoder = stepwise_encoder(3, t_h, t_f)
		# self.ego_var_encoder = st_encoder()
		# self.ego_mean_encoder = st_encoder()
		# self.ego_scale_encoder = st_encoder()
		# self.ego_intention_encoder = intention_encoder()
		# self.ego_similarity_encoder = similarity_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())
		self.intention_decoder = MLP(256*2, self.intention_dim, hid_feat=(256, 128), activation=nn.ReLU())
		self.similarity_decoder = MLP(256*2, self.angel_dim, hid_feat=(256, 128), activation=nn.ReLU())

	
	def forward(self, x, intention, past_similarity, mask=None):
		'''
			x: batch size, t_p, 6
			intention: batch size, t_p, 3
			past_similarity: batch size, t_p, 3
		'''
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		
		ego_var_embed = self.ego_var_encoder(x)
		ego_mean_embed = self.ego_mean_encoder(x)
		ego_scale_embed = self.ego_scale_encoder(x)
		ego_intention_embed = self.ego_intention_encoder(intention)
		ego_similarity_embed = self.ego_similarity_encoder(past_similarity)
		# B, 256
		# social_embed = self.old_social_encoder(x, mask)
		# social_input = ego_var_embed*torch.exp(ego_scale_embed/2)+ego_mean_embed
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		intention_social_embed = self.intention_social_encoder(x, mask)
		intention_social_embed = intention_social_embed.squeeze(1)
		similarity_social_embed = self.similarity_social_encoder(x, mask)
		similarity_social_embed = similarity_social_embed.squeeze(1)
		# B, 256

		intention_total = torch.cat((ego_intention_embed, intention_social_embed), dim=-1)
		guess_intention = self.intention_decoder(intention_total).reshape(x.size(0), self.n, self.fut_len, 3) # B, K, T, 1
		# guess_intention = torch.cat((torch.zeros_like(guess_intention[:, :, :,:1]), torch.zeros_like(guess_intention[:, :, :,:1]), guess_intention), dim=-1)
		
		similarity_total = torch.cat((ego_similarity_embed, similarity_social_embed), dim=-1)
		guess_similarity = self.similarity_decoder(similarity_total).reshape(x.size(0), self.n, self.fut_len, 1) # B, K, T, 1
		

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2) # B, T, 2
		
		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total) # B, 1

		guess_scale_feat = self.scale_encoder(guess_scale) 
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1) 
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2) # B, K, T, 2

		sample_3d = torch.cat((guess_var, torch.zeros_like(guess_var[:, :, :,:1])), dim=-1)
		# goal_sample = self.calculate_future_vectors_recursive(sample_3d, guess_intention, guess_similarity)
		goal_sample = self.solve_b_batch(sample_3d, guess_intention, guess_similarity) # B, K, T, 3
		goal_sample = torch.cat([sample_3d[:, :, 0:1, :], goal_sample[:, :, :-1, :]], dim=2)

		return guess_var, guess_mean, guess_scale, guess_intention, guess_similarity, goal_sample[..., :2]
	
	def calculate_future_vectors_recursive(self, x, intention, similarity):
		"""
		根据递推方式计算未来的 T 帧 b 向量。通过逐步计算每一帧的 b 向量，使用上一帧的向量作为下一帧的输入。
		
		参数:
		x (torch.Tensor): 观测轨迹，形状为 (B, M, T, 3)，B是batch size，M是多模态数，T是预测帧数，3表示坐标维度
		intention (torch.Tensor): 意图向量，形状为 (B, M, T, 3)，M是多模态数，T是预测帧数，3表示坐标维度
		similarity (torch.Tensor): 相似度向量，形状为 (B, M, T, 1)，M是多模态数，T是预测帧数，1表示标量
		
		返回:
		torch.Tensor: 预测向量 b，形状为 (B, M, T, 3)
		"""
		B, M, T, _ = x.shape  # 获取批次大小、多模态数和预测帧数
		future_vectors = []  # 存储每一帧的b向量
		
		# 初始化第一个a向量为x的第0帧
		a = x[:, :, 0, :]  # 获取x的第0帧，形状 (B, M, 3)
		
		for t in range(T-1):
			# 获取当前的意图向量和相似度向量
			c = intention[:, :, t, :]  # 意图向量，形状 (B, M, 3)
			d = similarity[:, :, t, :]  # 相似度标量，形状 (B, M, 1)
			
			# 计算当前帧的b向量
			b = self.solve_b_batch(a, c, d)  # 计算b向量，调用之前的solve_b_batch函数
			
			# 将计算出的b向量添加到未来轨迹列表
			future_vectors.append(b.unsqueeze(2))  # 添加到未来轨迹列表
			
			a =  b  # 更新a向量为当前帧的b向量
		
		# 将未来向量拼接成 (B, M, T-1, 3)
		future_vectors = torch.cat(future_vectors, dim=2)  # 形状为 (B, M, T-1, 3)
		
		# 插入x的第0帧到未来轨迹的最前面
		future_vectors = torch.cat([x[:, :, 0, :].unsqueeze(2), future_vectors], dim=2)  # 形状为 (B, M, T, 3)
		
		return future_vectors
	
	def solve_b_batch(self, a, c, d):
		"""
		完全向量化版本，用于批量计算 b 向量。

		参数:
		a (torch.Tensor): 向量 a，形状为 (B, M, T, 3)
		c (torch.Tensor): 向量 c，形状为 (B, M, T, 3)
		d (torch.Tensor): 向量 d，形状为 (B, M, T, 1)

		返回:
		torch.Tensor: 向量 b，形状为 (B, M, T, 3)
		"""
		# 避免 a 的模长为零
		eps = 1e-8
		a_dot_a = torch.sum(a * a, dim=-1, keepdim=True)
		a_dot_a = torch.clamp(a_dot_a, min=eps)  # 防止除零

		# 计算 λ 和垂直分量
		lambda_ = d / a_dot_a
		b_perpendicular = torch.cross(c, a, dim=-1) / a_dot_a

		# 合成 b 向量
		b = lambda_ * a + b_perpendicular

		return b

	



