import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear

from models.layers import PositionalEncoding, ConcatSquashLinear

class st_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 2
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 256
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X):
		'''
			X: b, T, 2

			return: b, F
		'''
		'''
			卷积层通常要求输入数据的维度是 [batch_size, channels, length]，其中：

			batch_size 是样本数，
			channels 是特征的数量，
			length 是序列的长度(在时间序列任务中，通常是时间步数 T)。
		'''
		X_t = torch.transpose(X, 1, 2)#这里的X_t是将X的维度转换为b, 2, T，T是时间维度
		X_after_spatial = self.relu(self.spatial_conv(X_t))#这里的X_after_spatial是将X_t经过一次卷积操作，输出的维度是b, 32, T，获取空间信息
		X_embed = torch.transpose(X_after_spatial, 1, 2)#这里的X_embed是将X_after_spatial的维度转换为b, T, 32

		output_x, state_x = self.temporal_encoder(X_embed)# 这里的output_x是经过GRU所有时间步的输出，维度是b, T, 256，state_x是GRU的最后一个状态，维度是1, b, 256
		state_x = state_x.squeeze(0)#这里的state_x是将state_x的维度转换为b, 256

		return state_x


class social_transformer(nn.Module):
	def __init__(self):
		super(social_transformer, self).__init__()
		self.encode_past = nn.Linear(60, 256, bias=False)
		# self.encode_past = nn.Linear(48, 256, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1) # 这里的h_feat是将h的维度转换为batch_size, 1, 256
		# print(h_feat.shape)
		# n_samples, 1, 64
		h_feat_ = self.transformer_encoder(h_feat, mask) # 这里的h_feat_是将h_feat经过transformer_encoder的输出，维度是batch_size, 1, 256
		h_feat = h_feat + h_feat_ # 残差,transformer_encoder 内部有残差连接，这里可能有些多余

		return h_feat


class TransformerDenoisingModel(Module):

	def __init__(self, context_dim=256, tf_layer=2):
		super().__init__()
		self.encoder_context = social_transformer() # 社交信息获取
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
		self.concat1 = ConcatSquashLinear(2, 2*context_dim, context_dim+3)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
		self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)


	def forward(self, x, beta, context, mask):
		'''
			调用样式：self.model(y, beta, x, mask)
			所以x是未来轨迹，beta是噪声强度，context是语义信息（社交信息或过去轨迹），mask是掩码
		'''
		batch_size = x.size(0)
		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
		
		x = self.concat1(ctx_emb, x)
		final_emb = x.permute(1,0,2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1,0,2)
		trans = self.concat3(ctx_emb, trans)
		trans = self.concat4(ctx_emb, trans)
		return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, x, beta, context, mask):
		batch_size = x.size(0)
		beta = beta.view(beta.size(0), 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask) # 社交编码
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)

		ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, 10, 1).unsqueeze(2) # (B, 10, 1, F+3) 10是

		x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, 20, 512) # (B*10, 20, 512)

		final_emb = x.permute(1, 0, 2) # (20, B*10, 512)
		final_emb = self.pos_emb(final_emb) # (20, B*10, 512)
		
		trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 10, 20, 512) # (B, 10, 20, 512)

		trans = self.concat3.batch_generate(ctx_emb, trans) # (B, 10, 20, 256)
		trans = self.concat4.batch_generate(ctx_emb, trans) # (B, 10, 20, 128)
		return self.linear.batch_generate(ctx_emb, trans) # (B, 10, 20, 2)
