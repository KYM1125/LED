import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder, simularity_encoder, simularity_social_transformer, intention_social_transformer, intention_encoder
from scipy.spatial.transform import Rotation as R

class StepweiseInitializer(nn.Module):
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
		super(StepweiseInitializer, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = 1 * d_f * k_pred
		self.intention_dim = 1 * 3 * k_pred
		self.similarity_dim = 1 * 1 * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer(t_h)
		self.intention_social_encoder = intention_social_transformer(t_h)
		self.simularity_social_encoder = simularity_social_transformer(t_h)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()
		self.ego_intention_encoder = intention_encoder()
		self.ego_simularity_encoder = simularity_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, 1 * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())
		self.intention_decoder = MLP(256*2, 1 * 9, hid_feat=(256, 128), activation=nn.ReLU())
		self.simularity_decoder = MLP(256*2, 1 * 3, hid_feat=(256, 128), activation=nn.ReLU())
		self.next_goal_expander = MLP(2, 6, hid_feat=(256, 128), activation=nn.ReLU())

	
	def forward(self, x, intention, past_similarity, mask=None):
		'''
			x: batch size, t_p, 6
			intention: batch size, t_p, 3
			past_similarity: batch size, t_p, 3
		'''
		# 初始化
		batch_size, t_p, _ = x.size()
		pred_len = self.fut_len  # 预测帧数
		pred_mean = []  
		pred_var = []  
		pred_intentions = []
		pred_simularities = []
		goal_mean = []  # 用于保存逐帧解码的结果

		# 初始的社交编码
		x_current = x  # 当前输入帧 (初始为观测帧)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

		for t in range(pred_len):
			# 编码
			social_embed = self.social_encoder(x_current, mask).squeeze(1)
			intention_social_embed = self.intention_social_encoder(intention, mask).squeeze(1)
			simularity_social_embed = self.simularity_social_encoder(past_similarity, mask).squeeze(1)

			# 自特征编码
			ego_var_embed = self.ego_var_encoder(x_current)
			ego_mean_embed = self.ego_mean_encoder(x_current)
			ego_scale_embed = self.ego_scale_encoder(x_current)
			ego_intention_embed = self.ego_intention_encoder(intention)
			ego_simularity_embed = self.ego_simularity_encoder(past_similarity)

			# 解码意图
			intention_total = torch.cat((ego_intention_embed, intention_social_embed), dim=-1)
			guess_intention = self.intention_decoder(intention_total).reshape(batch_size, 1, 9)  # B, 1, 9

			# 解码相似性
			simularity_total = torch.cat((ego_simularity_embed, simularity_social_embed), dim=-1)
			guess_simularity = self.simularity_decoder(simularity_total).reshape(batch_size, 1, 3)  # B, 1, 3

			# 解码均值和方差
			mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
			guess_mean = self.mean_decoder(mean_total).reshape(batch_size, 1, 2)  # B, 1, 2

			scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
			guess_scale = self.scale_decoder(scale_total)

			guess_scale_feat = self.scale_encoder(guess_scale)
			var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
			guess_var = self.var_decoder(var_total).reshape(batch_size, self.n, 1, 2)  # B, K, 1, 2

			# 生成下一帧目标点
			mean_3d = torch.cat((guess_mean, torch.zeros_like(guess_mean[:, :, :1])), dim=-1)
			next_goal = self.calculate_future_vectors_recursive(mean_3d, guess_intention[:,:,3:6], guess_simularity).unsqueeze(1)

			# 保存当前时间步解码的结果
			pred_var.append(guess_var)
			pred_mean.append(guess_mean)
			pred_intentions.append(guess_intention)
			pred_simularities.append(guess_simularity)
			goal_mean.append(next_goal[..., :2])  # 只取前两维 (x, y)

			# 更新输入帧 (添加解码结果作为新的观测帧)
			next_goal = self.next_goal_expander(next_goal[..., :2])

			x_next = torch.cat([x_current[:, 1:], next_goal[..., :6]], dim=1)
			intention = torch.cat([intention[:, 1:], guess_intention], dim=1)  # 更新意图
			past_similarity = torch.cat([past_similarity[:, 1:], guess_simularity], dim=1)  # 更新相似性
			x_current = x_next  # 更新当前输入帧
			

		# 拼接所有时间步的结果
		pred_var = torch.cat(pred_var, dim=2) # B, K, T, 2
		pred_mean = torch.cat(pred_mean, dim=1) # B, T, 2
		pred_intentions = torch.cat(pred_intentions, dim=1)  # B, T, 9
		pred_simularities = torch.cat(pred_simularities, dim=1)  # B, T, 3
		goal_mean = torch.cat(goal_mean, dim=1)  # B, T, 2

		return pred_var, pred_mean, guess_scale, pred_intentions[:,:,3:6], pred_simularities[:,:,1], goal_mean

	
	def calculate_future_vectors_recursive(self, x, intention, similarity):
		"""
		根据递推方式计算未来的 T 帧 b 向量。通过逐步计算每一帧的 b 向量，使用上一帧的向量作为下一帧的输入。
		
		参数:
		x (torch.Tensor): 观测轨迹，形状为 (B, 1, 3)，B是batch size，M是多模态数，T是预测帧数，3表示坐标维度
		intention (torch.Tensor): 意图向量，形状为 (B, 1, 3)，M是多模态数，T是预测帧数，3表示坐标维度
		similarity (torch.Tensor): 相似度向量，形状为 (B, 1, 1)，M是多模态数，T是预测帧数，1表示标量
		
		返回:
		torch.Tensor: 预测向量 b，形状为 (B, 3)
		"""
		# 初始化第一个a向量为x的第0帧
		a = x[:, 0, :]  # 获取x的第0帧，形状 (B, 3)
		
		# 获取当前的意图向量和相似度向量
		c = intention[:, 0, :]  # 意图向量，形状 (B, 3)
		d = similarity[:, 0, :]  # 相似度标量，形状 (B, 1)
		
		# 计算当前帧的b向量
		b = self.solve_b_batch(a, c, d)  # 计算b向量，调用之前的solve_b_batch函数
		
		return b


	def solve_b_batch(self, a, c, d):
		"""
		批量计算 b 向量，优化的版本使用矩阵运算避免逐个样本的计算。
		
		参数:
		a (torch.Tensor): 向量 a，形状为 (B, 1, 3)
		c (torch.Tensor): 向量 a 叉乘 b 的结果 c，形状为 (B, 1, 3)
		d (torch.Tensor): 向量 a 点乘 b 的结果 d，形状为 (B, 1, 1)
		
		返回:
		torch.Tensor: 向量 b，形状为 (B, 3)
		"""
		# 计算向量 a 和 c 的模长
		a_norm = torch.norm(a, dim=-1, keepdim=True)  # (B, 1, 1)
		c_norm = torch.norm(c, dim=-1, keepdim=True)  # (B, 1, 1)

		# 计算 b 的模长
		b_norm = torch.sqrt(c_norm**2 + d**2) / a_norm  # (B, 1, 1)

		# 计算叉积方向
		b_direction = torch.cross(a, c, dim=-1)  # (B, 1, 3)
		b_direction_norm = torch.norm(b_direction, dim=-1, keepdim=True)  # (B, 1, 1)
		b_direction = b_direction / b_direction_norm  # 归一化方向，(B, 1, 3)

		# 批量计算 b 向量
		b = b_norm * b_direction  # (B, 1, 3)

		return b
