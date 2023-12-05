import math

import numpy as np
import torch
import torch.nn as nn
import warnings


def get_pad_attention_mask(Q_seq, K_seq, pad_index_in_vocab):
    """
    获取针对pad符号的attention_mask
    :param Q_seq: [batch_size, q_seq_len]
    :param K_seq: [batch_size, k_seq_len]
    :param pad_index_in_vocab: pad在字典中的位置
    :return: attention_mask [batch_size, q_seq_len, k_seq_len],对于K_seq中的pad添加mask
    """
    # 获取batch_size和序列长度
    batch_size, q_seq_len = Q_seq.shape[0], Q_seq.shape[1]
    k_seq_len = K_seq.shape[1]

    # 根据pad在字典中索引构建注意力掩码
    # [batch_size, k_seq_len] -> [batch_size, 1, k_seq_len]
    pad_attention_mask = K_seq.detach().eq(pad_index_in_vocab).unsqueeze(1)
    # [batch_size, 1, k_seq_len] -> [batch_size, q_seq_len, k_seq_len]
    return pad_attention_mask.expand(batch_size, q_seq_len, k_seq_len)


def get_all_zero_attention_mask(batch_size, q_seq_len, k_seq_len):
    """
    构造全零注意力掩码
    :param batch_size: 批处理大小
    :param q_seq_len: q序列长度
    :param k_seq_len: k序列长度
    :return: attention_mask [batch_size, q_seq_len, k_seq_len]
    """
    # 构造全零注意力掩码
    all_zero_attention_mask = torch.zeros((batch_size, q_seq_len, k_seq_len), dtype=torch.bool)
    return all_zero_attention_mask


def get_subsequent_attention_mask(seq):
    """
    构造decoder时的屏蔽下文attention_mask
    :param seq: [batch_size, seq_len]
    :return: subsequence_mask [batch_size, seq_len, seq_len]
    """
    # [batch_size, seq_len, seq_len]
    attention_mask_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    # 生成一个严格上三角矩阵
    # np.triu()返回一个上三角矩阵，自对角线k以下元素全部置为0，k代表对角线上下偏移程度，这里将k设置为1是为了构建严格上三角矩阵（对角线全为0）
    subsequence_mask = np.triu(np.ones(attention_mask_shape), k=1)
    # 如果没转成byte，这里默认是Double(float64)，占据的内存空间大，浪费，用byte就够了
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class PositionalEncoding(nn.Module):
    """
    位置编码层
    """

    def __init__(self,
                 embedding_dim,
                 seq_len):
        """
        构造方法
        :param embedding_dim: 词向量维度
        :param seq_len: 序列长度
        """
        super(PositionalEncoding, self).__init__()
        # 以下为transformer的positionalEncoding公式
        # 第pos个词向量偶数位置： PE_{(pos, 2i)} = \sin (\frac{pos}{10000^{\frac{2i}{d_{k}}}})
        # 第pos个词向量奇数位置： PE_{(pos, 2i + 1)} = \cos (\frac{pos}{10000^{\frac{2i}{d_{k}}}})
        # 其中 \frac{1}{10000^{\frac{2i}{d_{k}}}} 可以等价替换为 e^{-\frac{2i}{d} ln(10000)}
        positional_encoding = torch.zeros(seq_len, embedding_dim)
        # 生成位置（这里添加一个维度在前面是为了后面批次计算）[seq_len] -> [seq_len, 1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # e^{-\frac{2i}{d} ln(10000)} shape: [embedding_dim / 2]
        div_term = torch.exp(-(torch.arange(0, embedding_dim, 2).float() / embedding_dim) * (math.log(10000.0)))
        # 偶数部分
        # position[seq_len, 1] * div_term[embedding_dim / 2]
        # -> 广播机制变为 [seq_len, embedding_dim / 2] /dot [seq_len, embedding_dim / 2]
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # 奇数部分
        if embedding_dim % 2 == 0:
            positional_encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            positional_encoding[:, 1::2] = torch.cos(position * div_term[0:-1])

        # 定义为固定参数
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, seq):
        # 获取当前序列长度，并叠加上位置编码
        # 假定这里x输入的shape为 [batch_size, seq_len, embedding_dim]
        # 广播机制self.positional_encoding shape变为[batch_size, seq_len, embedding_dim]
        return seq + self.positional_encoding[:seq.shape[1], :]


class ScaledDotProductAttention(nn.Module):
    """
    点积缩放
    softmax(\frac{Q @ K^T}{ sqrt{d_k}}) @ V
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attention_mask):
        """
        前向传播
        softmax(\frac{Q @ K^T}{ sqrt{d_k}}) @ V
        :param Q: Q: Query [batch_size, num_heads, q_seq_len, q_dim]
        :param K: K: Key [batch_size, num_heads, k_seq_len, k_dim]
        :param V: V: Value [batch_size, num_heads, v_seq_len, v_dim]
        :param attention_mask: [batch_size, num_heads, q_seq_len, k_seq_len]
        :return: (context: [batch_size, num_heads, seq_len, v_dim],
                    attention: [batch_size, num_heads, q_seq_len, k_seq_len])
        """
        # 输入数据校验
        # 由于要进行Q @　K^T， 因此q_dim必须等于k_dim
        assert Q.shape[-1] == K.shape[-1], 'Q, K输入词向量维度必须相同！'
        # Q @ K^T 得到的结果shape为[batch_size, num_heads, q_seq_len, k_seq_len]，后续再与V做矩阵乘法，
        # 因此k_seq_len必须等于v_seq_dim
        assert K.shape[-1] == V.shape[-1], 'K, V输入的序列长度必须相同！'
        # 由于attention_mask作用在 Q @ K^T 上，因此要求attention_mask尺寸必须是[batch_size, num_heads, q_seq_len, k_seq_len]
        assert Q.shape[-2] == attention_mask.shape[-2] and K.shape[-2] == attention_mask.shape[-1], \
            f'attention_mask尺寸{attention_mask.shape} != ' \
            f'Q @　K^T 的尺寸{[Q.shape[0], Q.shape[1], Q.shape[2], K.shape[-2]]}!'

        # 获取词向量维度做缩放
        d_k = Q.shape[-1]
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 添加mask，将mask位置设置为无限小，通过softmax基本就是0，最后打掩码位置就不会对结果产生影响
        scores.masked_fill_(attention_mask, -1e9)
        # softmax(\frac{Q @ K^T}{ sqrt{d_k}})
        attention = self.softmax(scores)
        # softmax(\frac{Q @ K^T}{ sqrt{d_k}}) @ V
        context = torch.matmul(attention, V)
        # 返回结果和注意力矩阵
        return context, attention


class MultiHeadSelfAttention(nn.Module):
    """
    多头注意力层
    """

    def __init__(self,
                 embedding_dim,
                 k_dim,
                 v_dim,
                 num_heads):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param k_dim: 最后输出的K的维度
        :param v_dim: 最后输出的V的维度
        :param num_heads: 总共要几个头
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        # 定义生成QKV矩阵的线性层
        # 注意这里考虑多头注意力，因此实际输出向量长度为原维度的num_heads倍，后面再拆分
        self.W_Q = nn.Linear(embedding_dim, k_dim * num_heads)
        self.W_K = nn.Linear(embedding_dim, k_dim * num_heads)
        self.W_V = nn.Linear(embedding_dim, v_dim * num_heads)
        # 多头结果拼接融合层
        self.fc = nn.Linear(v_dim * num_heads, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # ScaledDotProductAttention
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, attention_mask):
        """
        前向传播
        计算
        :param Q: Query [batch_size, q_seq_len, q_embedding_dim]
        :param K: Key [batch_size, k_seq_len, k_embedding_dim]
        :param V: Value [batch_size, v_seq_len, v_embedding_dim]
        :param attention_mask: 掩码，用来标记padding等 [batch_size, q_seq_len, k_seq_len]
        :return: (self-attention结果: [batch_size, seq_len, embedding_dim],
                    attention: [batch_size, num_heads, q_seq_len, k_seq_len])
        """
        assert Q.shape[-1] == K.shape[-1], 'Q, K输入词向量维度必须相同！'

        # 获取残差连接输入和batch_size
        residual, batch_size = Q, Q.shape[0]

        # 得到QKV，并拆分为多头[batch_size, seq_len, num_heads, k or v dim]
        # 然后transpose成[batch_size, num_heads, seq_len, k or v dim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_heads, self.v_dim).transpose(1, 2)

        # attention_mask
        # [batch_size, seq_len, k_dim] -> [batch_size, 1, seq_len, k_dim] -> [batch_size, num_heads, seq_len, k_dim]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # 通过ScaledDotProductAttention聚合上下文信息
        context, attention = self.scaled_dot_product_attention(q_s, k_s, v_s, attention_mask)
        # 首先通过transpose方法转置 [batch_size, num_heads, seq_len, v_dim] -> [batch_size, seq_len, num_heads, v_dim]
        # 然后通过contiguous解决转置带来的非连续存储问题，提升性能
        # 之后再用将多头信息concat到一起[batch_size, seq_len, num_heads, v_dim] -> [batch_size, seq_len, num_heads * v_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.v_dim)
        # 线性变换融合多头信息 [batch_size, seq_len, num_heads * v_dim] -> [batch_size, seq_len, embedding_dim]
        output = self.fc(context)
        # Add and LayerNorm
        output = self.layer_norm(residual + output)
        return output, attention


class PositionWiseFeedForward(nn.Module):
    """
    前馈神经网络
    Position-wise意为对每个点独立做，即对序列中的每个token独立过同一个MLP，即作用在输入的最后一个维度上
    这里可选Conv1D和Linear两种实现方式，Linear考虑全局，而Conv1D则是
    """

    def __init__(self, input_size,
                 hidden_size,
                 mode: str):
        """
        构造方法
        :param input_size: 词向量维度
        :param hidden_size: 隐藏层维度
        :param mode: 使用 linear or conv
        """
        super(PositionWiseFeedForward, self).__init__()
        assert mode in ['linear', 'conv'], "mode 必须是 'linear' or 'conv'!\a"
        self.__mode = mode
        if self.__mode == 'linear':
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(in_features=hidden_size, out_features=input_size, bias=False))
        elif self.__mode == 'conv':
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=1, bias=False)
            )
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        """
        前向传播
        :param x: [batch_size, seq_len, embedding_dim]

        :return: x + W @ x
        """
        residual = x
        if self.__mode == 'linear':
            return self.layer_norm(residual + self.fc(x))
        elif self.__mode == 'conv':
            # x 需要转置一下，让卷积在seq_len维度进行卷积
            return self.layer_norm(residual + self.conv(x.transpose(1, 2)).transpose(1, 2))
        else:
            raise ValueError("mode 必须是 'linear' or 'conv'!\a")


class EncodeLayer(nn.Module):
    """
    Encoder块
    包含自注意力层和前馈神经网络层
    """

    def __init__(self,
                 embedding_dim,
                 k_dim,
                 v_dim,
                 num_heads,
                 ffn_hidden_size,
                 ffn_mode):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param k_dim: 最后输出的K的维度
        :param v_dim: 最后输出的V的维度
        :param ffn_hidden_size: 前馈神经网络的隐藏层神经元结点个数
        :param ffn_mode: 前馈神经网络的模式，可选 'linear' 和 'conv'
        :param num_heads: 总共要几个头
        """
        super(EncodeLayer, self).__init__()
        # 实例化多头注意力层
        self.encode_self_attention = MultiHeadSelfAttention(embedding_dim, k_dim, v_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, ffn_hidden_size, mode=ffn_mode)

    def forward(self, inputs, attention_mask):
        output, attention = self.encode_self_attention(Q=inputs, K=inputs, V=inputs, attention_mask=attention_mask)
        output = self.feed_forward(output)

        return output, attention


class Encoder(nn.Module):
    """
    Encoder
    注意这里可选是否需要embedding，需要embedding时请指定字典长度vocab_size和pad符在字典中的索引pad_index_in_vocab
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 seq_len: int,
                 num_layers: int,
                 k_dim: int = 0,
                 v_dim: int = 0,
                 ffn_hidden_size: int = 0,
                 need_embedding: bool = True,
                 vocab_size: int = 0,
                 pad_index_in_vocab: int = 0,
                 ffn_mode: str = 'linear',
                 no_pad: bool = False):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param num_heads: 多头注意力头数
        :param seq_len: 序列长度
        :param num_layers: encoder块的层数
        :param k_dim: 单头自注意力key的维度
        :param v_dim: 单头自注意力value的维度
        :param ffn_hidden_size: 前馈神经网络隐藏层大小
        :param need_embedding: 是否需要embedding
        :param vocab_size: 词典长度（只有在need_embedding=True时生效）
        :param pad_index_in_vocab: pad符在字典中的index（只有在need_embedding=True时生效）
        :param no_pad: 输入序列没有pad，模型将自动生成全0mask，默认为False（need_embedding为True时不生效）
        :param ffn_mode: 前馈神经网络模式，默认为线性
        """
        super(Encoder, self).__init__()
        # 检验参数合理性
        if need_embedding and vocab_size <= 0:
            raise ValueError(f"'need_embedding'为True时，必须要设置合法的字典长度'vocab_size'!\a")
        if need_embedding is False and vocab_size != 0:
            warnings.warn(f"'need_embedding'为True时，字典长度'vocab_size'将不生效", UserWarning)

        assert (need_embedding and vocab_size != 0) or (need_embedding is False and vocab_size == 0)
        self.__need_embedding = need_embedding
        self.no_pad = no_pad
        self.padding_index_in_vocab = pad_index_in_vocab
        # 如果没有给定前馈神经网络的隐藏层参数就直接等于embedding_dim
        if ffn_hidden_size == 0:
            ffn_hidden_size = embedding_dim
        if k_dim == 0:
            k_dim = embedding_dim
        if v_dim == 0:
            v_dim = embedding_dim

        # 判断是否需要对输入数据进行embedding操作
        if need_embedding:
            # 将输入的单个字典索引数值变为长度为embedding_dim的向量
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置编码层
        self.pos_embedding = PositionalEncoding(embedding_dim, seq_len)
        # 创建encoder块
        self.encoder_layers = nn.ModuleList([EncodeLayer(embedding_dim=embedding_dim,
                                                         k_dim=k_dim,
                                                         v_dim=v_dim,
                                                         num_heads=num_heads,
                                                         ffn_hidden_size=ffn_hidden_size,
                                                         ffn_mode=ffn_mode) for _ in range(num_layers)])

    def forward(self, x, attention_mask=None):
        """
        前向传播
        :param x: [batch_size, seq_len] or [batch_size, seq_len, embedding_dim]
        :param attention_mask: 需要打上掩码的部分[batch_size, seq_len, seq_len]
        :return: (output: [batch_size, seq_len, embedding_dim])
        """
        # embedding
        if self.__need_embedding:
            # 获取注意力掩码
            attention_mask = get_pad_attention_mask(x, x, self.padding_index_in_vocab)
            # embedding
            x = self.embedding(x)
            if self.no_pad is True:
                warnings.warn("设置need_embedding为True时，no_pad将不生效！", UserWarning)
        else:
            if self.no_pad:
                # 设置全0mask
                attention_mask = get_all_zero_attention_mask(x.shape[0], x.shape[1], x.shape[1])
            else:
                if attention_mask is None:
                    raise ValueError(
                        f"当实例化encoder时指定了need_embedding为False且no_pad也为False时，必须输入attention_mask!")

        # 叠加位置编码
        out_put = self.pos_embedding(x)

        # 创建各层注意力权重列表
        encoder_self_attentions = []

        # 开始输入encoder块
        for encoder_layer in self.encoder_layers:
            # 依次丢进encoder块中
            out_put, attention = encoder_layer(out_put, attention_mask)
            # 将注意力权重加入列表保存
            encoder_self_attentions.append(attention)

        return out_put, encoder_self_attentions


class DecoderLayer(nn.Module):
    """
    Decoder块
    包含自注意力层、交叉注意力层和前馈神经网络层
    """

    def __init__(self,
                 embedding_dim,
                 k_dim,
                 v_dim,
                 num_heads,
                 ffn_hidden_size,
                 ffn_mode):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param k_dim: 最后输出的K的维度
        :param v_dim: 最后输出的V的维度
        :param ffn_hidden_size: 前馈神经网络的隐藏层神经元结点个数
        :param ffn_mode: 前馈神经网络的模式，可选 'linear' 和 'conv'
        :param num_heads: 总共要几个头
        """
        super(DecoderLayer, self).__init__()
        # 实例化多头自注意力层
        self.decoder_self_attention = MultiHeadSelfAttention(embedding_dim, k_dim, v_dim, num_heads)
        # 实例化交叉注意力层
        self.decoder_encoder_attention = MultiHeadSelfAttention(embedding_dim, k_dim, v_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, ffn_hidden_size, mode=ffn_mode)

    def forward(self, decoder_input, encoder_output, dec_self_attention_mask, dec_enc_attention_mask):
        """
        前向传播
        :param decoder_input: decoder的输入
        :param encoder_output: encoder的输出
        :param dec_self_attention_mask: decoder自注意力的注意力掩码
        :param dec_enc_attention_mask: decoder-encoder交叉注意力的注意力掩码
        :return: (decoder_output: [batch_size, seq_len, embedding_dim],
                    decoder_self_attention: [batch_size, num_heads, q_seq_len, k_seq_len],
                    decoder_encoder_attention: [batch_size, num_heads, q_seq_len, k_seq_len])
        """
        # 掩码自注意力
        decoder_output, decoder_self_attention = self.decoder_self_attention(Q=decoder_input,
                                                                             K=decoder_input,
                                                                             V=decoder_input,
                                                                             attention_mask=dec_self_attention_mask)
        # 交叉注意力
        decoder_output, decoder_encoder_attention = \
            self.decoder_encoder_attention(Q=decoder_output,
                                           K=encoder_output,
                                           V=encoder_output,
                                           attention_mask=dec_enc_attention_mask)
        # 前馈神经网络
        decoder_output = self.feed_forward(decoder_output)

        return decoder_output, decoder_self_attention, decoder_encoder_attention


class Decoder(nn.Module):
    """
    Encoder
    注意这里可选是否需要embedding，需要embedding时请指定字典长度vocab_size和pad符在字典中的索引pad_index_in_vocab
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 seq_len: int,
                 num_layers: int,
                 k_dim: int = 0,
                 v_dim: int = 0,
                 ffn_hidden_size: int = 0,
                 need_embedding: bool = True,
                 vocab_size: int = 0,
                 pad_index_in_vocab: int = 0,
                 ffn_mode: str = 'linear',
                 no_pad: bool = False):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param num_heads: 多头注意力头数
        :param seq_len: 序列长度
        :param num_layers: encoder块的层数
        :param k_dim: 单头注意力key的维度
        :param v_dim: 单头注意力value的维度
        :param ffn_hidden_size: 前馈神经网络隐藏层大小
        :param need_embedding: 是否需要embedding
        :param vocab_size: 词典长度（只有在need_embedding=True时生效）
        :param pad_index_in_vocab: pad符在字典中的index（只有在need_embedding=True时生效）
        :param ffn_mode: 前馈神经网络模式，默认为线性
        :param no_pad: 输入序列没有pad，模型将自动生成全0mask，默认为False（need_embedding为True时不生效）
        """
        super(Decoder, self).__init__()
        # 检验参数合理性
        if need_embedding and vocab_size <= 0:
            raise ValueError(f"'need_embedding'为True时，必须要设置合法的字典长度'vocab_size'!\a")
        if need_embedding is False and vocab_size != 0:
            warnings.warn(f"'need_embedding'为True时，字典长度'vocab_size'将不生效", UserWarning)
        assert (need_embedding and vocab_size is not None) or (need_embedding is False and vocab_size is None)
        self.__need_embedding = need_embedding
        self.no_pad = no_pad
        self.pad_index_in_vocab = pad_index_in_vocab
        # 如果没有给定前馈神经网络的隐藏层参数就直接等于embedding_dim
        if ffn_hidden_size == 0:
            ffn_hidden_size = embedding_dim
        if k_dim == 0:
            k_dim = embedding_dim
        if v_dim == 0:
            v_dim = embedding_dim

        # 判断是否需要对输入数据进行embedding操作
        if need_embedding:
            # 将输入的单个字典索引数值变为长度为embedding_dim的向量
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置编码层
        self.pos_embedding = PositionalEncoding(embedding_dim, seq_len)
        # 创建decoder块
        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_dim=embedding_dim,
                                                          k_dim=k_dim,
                                                          v_dim=v_dim,
                                                          num_heads=num_heads,
                                                          ffn_hidden_size=ffn_hidden_size,
                                                          ffn_mode=ffn_mode) for _ in range(num_layers)])

    def forward(self,
                decoder_input,
                encoder_input,
                encoder_output,
                decoder_self_attention_pad_mask=None,
                decoder_encoder_attention_pad_mask=None):
        """
        前向传播
        :param decoder_input: decoder输入
        :param encoder_input: encoder输入
        :param encoder_output: encoder输出
        :param decoder_self_attention_pad_mask:
                decoder自注意力掩码 [batch_size, decoder_input_seq_len, decoder_input_seq_len]
        :param decoder_encoder_attention_pad_mask:
                decoder_encoder交叉注意力掩码 [batch_size, decoder_input_seq_len, encoder_input_seq_len]
        :return: (decoder_output: [batch_size, decoder_seq_len, embedding_dim],
                    decoder_self_attentions: list[[batch_size, num_heads, decoder_seq_len, decoder_seq_len]],
                    decoder_encoder_attentions: list[[batch_size, num_heads, decoder_seq_len, encoder_seq_len]])
        """
        # embedding
        if self.__need_embedding:
            # 获取pad注意力掩码
            decoder_self_attention_pad_mask = \
                get_pad_attention_mask(decoder_input, decoder_input, self.pad_index_in_vocab)

            # 对decoder输入做embedding
            decoder_input = self.embedding(decoder_input)

            # 获取交叉注意力层掩码（获取encoder的pad位置）
            decoder_encoder_attention_pad_mask = \
                get_pad_attention_mask(decoder_input, encoder_input, self.pad_index_in_vocab)

            if self.no_pad is True:
                warnings.warn("设置need_embedding为True时，no_pad将不生效！", UserWarning)

        else:
            if self.no_pad:
                # 设置全0mask
                decoder_self_attention_pad_mask = get_all_zero_attention_mask(decoder_input.shape[0],
                                                                              decoder_input.shape[1],
                                                                              decoder_input.shape[1])
                # 设置全0mask
                decoder_encoder_attention_pad_mask = get_all_zero_attention_mask(decoder_input.shape[0],
                                                                                 decoder_input.shape[1],
                                                                                 encoder_input.shape[1])
            else:
                if decoder_self_attention_pad_mask is None or decoder_encoder_attention_pad_mask is None:
                    raise ValueError(f"当实例化encoder时指定了need_embedding为False且no_pad也设置为False时，"
                                     f"必须输入decoder_self_attention_pad_mask和decoder_encoder_attention_pad_mask!")

        # 获取decoder的下文注意力掩码
        decoder_self_attention_subsequent_mask = get_subsequent_attention_mask(decoder_input)
        # 叠加两个掩码，为了防止数值溢出，使用torch.gt方法固定为布尔类型
        decoder_self_attention_mask = \
            torch.gt((decoder_self_attention_pad_mask + decoder_self_attention_subsequent_mask), 0)

        # decoder输入叠加位置编码
        decoder_output = self.pos_embedding(decoder_input)

        # 创建decoder各层自注意力权重列表
        decoder_self_attentions = []
        # 创建decoder各层交叉注意力权重列表
        decoder_encoder_attentions = []

        # 开始输入decoder块
        for decoder_layer in self.decoder_layers:
            # 依次丢进decoder块中
            decoder_output, decoder_self_attention, decoder_encoder_attention = \
                decoder_layer(decoder_output,
                              encoder_output,
                              decoder_self_attention_mask,
                              decoder_encoder_attention_pad_mask)
            # 将注意力权重加入列表保存
            decoder_self_attentions.append(decoder_self_attention)
            decoder_encoder_attentions.append(decoder_encoder_attention)

        return decoder_output, decoder_self_attentions, decoder_encoder_attentions


class Transformer(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 encoder_seq_len: int,
                 decoder_seq_len: int,
                 encoder_num_layers: int,
                 decoder_num_layers: int,
                 result_size: int,
                 k_dim: int = 0,
                 v_dim: int = 0,
                 ffn_hidden_size: int = 0,
                 need_embedding: bool = False,
                 encoder_vocab_size: int = 0,
                 decoder_vocab_size: int = 0,
                 pad_index_in_vocab: int = 0,
                 ffn_mode: str = 'linear',
                 no_pad: bool = True):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param num_heads: 多头注意力头数
        :param encoder_seq_len: encoder序列长度
        :param decoder_seq_len: decoder序列长度
        :param encoder_num_layers: encoder块的层数
        :param decoder_num_layers: decoder块的层数
        :param result_size: 最后输出结果的长度
        :param k_dim: 单头注意力key的维度
        :param v_dim: 单头注意力value的维度
        :param ffn_hidden_size: 前馈神经网络隐藏层大小
        :param need_embedding: 是否需要embedding
        :param encoder_vocab_size: encoder词典长度（只有在need_embedding=True时生效）
        :param decoder_vocab_size: decoder词典长度（只有在need_embedding=True时生效）
        :param pad_index_in_vocab: pad符在字典中的index（只有在need_embedding=True时生效）
        :param ffn_mode: 前馈神经网络模式，默认为线性
        :param no_pad: 输入序列没有pad，模型将自动生成全0mask，默认为False（need_embedding为True时不生效）
        """
        super(Transformer, self).__init__()
        self.decoder_seq_len = decoder_seq_len
        self.embedding_dim = embedding_dim
        # 实例化encoder
        self.encoder = Encoder(embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               seq_len=encoder_seq_len,
                               num_layers=encoder_num_layers,
                               k_dim=k_dim,
                               v_dim=v_dim,
                               ffn_hidden_size=ffn_hidden_size,
                               need_embedding=need_embedding,
                               vocab_size=encoder_vocab_size,
                               pad_index_in_vocab=pad_index_in_vocab,
                               ffn_mode=ffn_mode,
                               no_pad=no_pad)
        # 实例化decoder
        self.decoder = Decoder(embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               seq_len=decoder_seq_len,
                               num_layers=decoder_num_layers,
                               k_dim=k_dim,
                               v_dim=v_dim,
                               ffn_hidden_size=ffn_hidden_size,
                               need_embedding=need_embedding,
                               vocab_size=decoder_vocab_size,
                               pad_index_in_vocab=pad_index_in_vocab,
                               ffn_mode=ffn_mode,
                               no_pad=no_pad)

        self.fc = nn.Linear(embedding_dim, result_size)

    def forward(self,
                encoder_input,
                decoder_input=None,
                encoder_attention_pad_mask=None,
                decoder_self_attention_pad_mask=None,
                decoder_encoder_attention_pad_mask=None):
        """
        前向传播
        :param encoder_input: encoder输入
        :param decoder_input: decoder输入
        :param encoder_attention_pad_mask: encoder自注意力掩码
        :param decoder_self_attention_pad_mask: decoder自注意力掩码
        :param decoder_encoder_attention_pad_mask: decoder-encoder交叉注意力掩码
        :return:
        """
        if self.training and decoder_input is None:
            raise ValueError("训练模式下，decoder的输入不能为None！")

        encoder_output, encoder_self_attentions = self.encoder(encoder_input, encoder_attention_pad_mask)

        decoder_self_attentions = decoder_encoder_attentions = None
        # 如果是训练
        if self.training:
            decoder_output, decoder_self_attentions, decoder_encoder_attentions = \
                self.decoder(decoder_input,
                             encoder_input,
                             encoder_output,
                             decoder_self_attention_pad_mask,
                             decoder_encoder_attention_pad_mask)
        else:
            if decoder_input is not None:
                warnings.warn("评估模式下，decoder_input不生效！", UserWarning)

            # 创建decoder第一个输入，全0作为开始符 [batch_size, seq_len, embedding_dim]
            decoder_output = torch.zeros((encoder_input.shape[0], 1, self.embedding_dim))
            # 开始遍历时间进行预测
            for i in range(self.decoder_seq_len):
                decoder_output_temp, decoder_self_attentions, decoder_encoder_attentions = \
                    self.decoder(
                        decoder_output,
                        encoder_input,
                        encoder_output,
                        decoder_self_attention_pad_mask,
                        decoder_encoder_attention_pad_mask)
                # 时间步上拼接
                decoder_output = torch.cat((decoder_input, decoder_output_temp[:, -1, :].unsqueeze(1)), dim=1)
            # 将第一个作为开始符的词向量丢弃
            decoder_output = decoder_output[:, 1:, :]
        results = self.fc(decoder_output)

        return results, (encoder_self_attentions, decoder_self_attentions, decoder_encoder_attentions)


def make_batch(sentences, src_vocab, tgt_vocab):
    # 把文本转成词表索引
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    # 把索引转成tensor类型
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


if __name__ == '__main__':
    # 定义句子输入
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    # 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)
    # encoder输入序列长度
    src_len = 5
    # decoder输入序列长度
    target_len = 5

    # 词向量维度
    embed_dim = 512
    # qkv矩阵长度
    dim_k = dim_v = 64
    # ffn的隐藏层结点个数
    dim_ffn = 2048
    n_layers = 6
    n_heads = 8

    model = Transformer(embedding_dim=embed_dim,
                        num_heads=n_heads,
                        encoder_num_layers=n_layers,
                        decoder_num_layers=n_layers,
                        encoder_seq_len=src_len,
                        decoder_seq_len=target_len,
                        result_size=tgt_vocab_size,
                        k_dim=dim_k,
                        v_dim=dim_v,
                        encoder_vocab_size=src_vocab_size,
                        decoder_vocab_size=tgt_vocab_size,
                        need_embedding=True)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences, src_vocab, tgt_vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs, (enc_self_attns, dec_self_attns, dec_enc_attns) = model(enc_inputs, dec_inputs)
        # output:[batch_size, tgt_len, tgt_vocab_size]
        loss = criterion(outputs.view(1 * 5, 7), target_batch.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
