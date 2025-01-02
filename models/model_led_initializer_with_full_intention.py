import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder, similarity_encoder, similarity_social_transformer, intention_social_transformer, intention_encoder, mode_transformer
import torch.nn.functional as F

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
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()
		self.ego_intention_encoder = intention_encoder()
		self.ego_similarity_encoder = similarity_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())
		self.intention_decoder = MLP(256*2, self.intention_dim, hid_feat=(256, 128), activation=nn.ReLU())
		self.similarity_decoder = MLP(256*2, self.angel_dim, hid_feat=(256, 128), activation=nn.ReLU())

		self.m2m_refine_attn_layer = mode_transformer(k_pred, t_f)
		self.to_conc = MLP(128, t_f, hid_feat=(64,32), activation=nn.ReLU())
		self.to_prob = MLP(128, 1, hid_feat=(64,32), activation=nn.ReLU())

	
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
		guess_intention = self.intention_decoder(intention_total).contiguous().view(x.size(0), self.n, self.fut_len, 3) # B, K, T, 3
		
		similarity_total = torch.cat((ego_similarity_embed, similarity_social_embed), dim=-1)
		guess_similarity = self.similarity_decoder(similarity_total).contiguous().view(x.size(0), self.n, self.fut_len, 1) # B, K, T, 1
		
		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total) # B, 1

		guess_scale_feat = self.scale_encoder(guess_scale) 
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1) 
		guess_var = self.var_decoder(var_total).contiguous().view(x.size(0), self.n, self.fut_len, 2) # B, K, T, 2

		sample_3d = torch.cat((guess_var, torch.zeros_like(guess_var[:, :, :,:1])), dim=-1)
		goal_var = self.calculate_future_vectors_recursive(sample_3d, guess_intention, guess_similarity)

		refined_mode = self.m2m_refine_attn_layer(goal_var) # B, K, 128
		conc = self.to_conc(refined_mode).unsqueeze(-1) 
		conc = 1.0 / (F.elu_(conc) + 1.0 + 0.02) # B, K, T, 1
		prob = self.to_prob(refined_mode).squeeze(-1) # B, K
		prob = F.softmax(prob, dim=-1)
		best_mode = torch.argmax(prob, dim=-1)
		best_intention = guess_intention[torch.arange(guess_intention.size(0), device=guess_intention.device), best_mode]
		best_similarity = guess_similarity[torch.arange(guess_similarity.size(0), device=guess_similarity.device), best_mode]

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2) # B, T, 2
		guess_mean_3d = torch.cat((guess_mean, torch.zeros_like(guess_mean[:, :, :1])), dim=-1)
		goal_mean = self.calculate_future_vectors_recursive(guess_mean_3d, best_intention, best_similarity)
		
		return {
			'guess_var': guess_var,
			'guess_mean': guess_mean,
			'guess_scale': guess_scale,
			'guess_intention': guess_intention,
			'guess_similarity': guess_similarity,
			'goal_var': goal_var[..., :2],
			'goal_mean': goal_mean[..., :2],
			'conc': conc,
			'prob': prob
		}
	
	def calculate_future_vectors_recursive(self, x, intention, similarity):
		"""
		根据递推方式计算未来的 T 帧 b 向量。通过逐步计算每一帧的 b 向量，使用上一帧的向量作为下一帧的输入。

		参数:
		x (torch.Tensor): 观测轨迹，形状为 (B, M, T, 3) 或 (B, T, 3)
		intention (torch.Tensor): 意图向量，形状为 (B, M, T, 3) 或 (B, T, 3)
		similarity (torch.Tensor): 相似度向量，形状为 (B, M, T, 1) 或 (B, T, 1)

		返回:
		torch.Tensor: 预测向量 b，形状为 (B, M, T, 3) 或 (B, T, 3)
		"""
		# 检查是否存在 M 维度
		has_m_dim = x.dim() == 4

		if has_m_dim:
			B, M, T, _ = x.shape  # 获取批次大小、多模态数和预测帧数
			future_vectors = []  # 存储每一帧的 b 向量
			a = x[:, :, 0, :]  # 初始化第一个 a 向量，形状 (B, M, 3)
		else:
			B, T, _ = x.shape  # 获取批次大小和预测帧数
			future_vectors = []  # 存储每一帧的 b 向量
			a = x[:, 0, :]  # 初始化第一个 a 向量，形状 (B, 3)

		for t in range(1, T):
			# 获取当前的意图向量和相似度向量
			c = intention[:, :, t, :] if has_m_dim else intention[:, t, :]  # 意图向量
			d = similarity[:, :, t, :] if has_m_dim else similarity[:, t, :]  # 相似度标量

			# 计算当前帧的 b 向量
			b = self.solve_b_batch(a, c, d)  # 调用solve_b_batch 函数

			# 将计算出的 b 向量添加到未来轨迹列表
			future_vectors.append(b.unsqueeze(2 if has_m_dim else 1))  # 动态调整维度

			# 更新 a 向量为当前计算出的 b 向量
			a = x[:, :, t, :] if has_m_dim else x[:, t, :]  # 更新 a 向量

		# 拼接未来向量
		dim_to_concat = 2 if has_m_dim else 1
		future_vectors = torch.cat(future_vectors, dim=dim_to_concat)  # 形状为 (B, M, T-1, 3) 或 (B, T-1, 3)

		# 插入 x 的第 0 帧到未来轨迹的最前面
		first_frame = x[:, :, 0, :].unsqueeze(dim_to_concat) if has_m_dim else x[:, 0, :].unsqueeze(dim_to_concat)
		future_vectors = torch.cat([first_frame, future_vectors], dim=dim_to_concat)  # 形状为 (B, M, T, 3) 或 (B, T, 3)

		return future_vectors



	def solve_b_batch(self, a, c, d):
		"""
		批量计算 b 向量，优化的版本使用矩阵运算避免逐个样本的计算。
		
		参数:
		a (torch.Tensor): 向量 a，形状为 (B, M, T, 3)
		c (torch.Tensor): 向量 a 叉乘 b 的结果 c，形状为 (B, M, T, 3)
		d (torch.Tensor): 向量 a 点乘 b 的结果 d，形状为 (B, M, T, 1)
		
		返回:
		torch.Tensor: 向量 b，形状为 (B, M, T, 3)
		"""
		# 计算向量 a 和 c 的模长
		a_norm = torch.norm(a, dim=-1, keepdim=True)  # (B, M, T, 1)
		c_norm = torch.norm(c, dim=-1, keepdim=True)  # (B, M, T, 1)

		# 计算 b 的模长
		b_norm = torch.sqrt(c_norm**2 + d**2) / a_norm  # (B, M, T, 1)

		# 计算叉积方向
		b_direction = torch.cross(a, c, dim=-1)  # (B, M, T, 3)
		b_direction_norm = torch.norm(b_direction, dim=-1, keepdim=True)  # (B, M, T, 1)
		b_direction = b_direction / b_direction_norm  # 归一化方向，(B, M, T, 3)

		# 批量计算 b 向量
		b = b_norm * b_direction  # (B, M, T, 3)

		return b

	



