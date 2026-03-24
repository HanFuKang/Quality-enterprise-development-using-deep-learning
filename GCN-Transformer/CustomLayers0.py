from keras.layers import Layer,Reshape,Permute,LayerNormalization,Dropout,Dense,LSTM
import tensorflow as tf
import numpy as np

class PE_original(Layer):
    """
    一个MxN的矩阵，可计算每个位置的位置情况
    """
    
    def __init__(self,emd_dim:int,max_seqL=256,scale_v=10000):
        self.emd_dim = emd_dim
        self.max_seqL = max_seqL
        self.scale_v = scale_v
        super(PE_original,self).__init__()


    def build(self, input_shape):
        self.encoding = np.zeros((self.max_seqL,self.emd_dim))
        self.pos = np.arange(0,self.max_seqL)
        self.pos = np.expand_dims(self.pos,axis=-1)
        self._2i = np.arange(0,self.emd_dim,step=2)
        self.encoding[:,0::2] = np.sin(self.pos / (self.scale_v ** (self._2i / self.emd_dim)))
        self.encoding[:,1::2] = np.cos(self.pos / (self.scale_v ** (self._2i / self.emd_dim)))
        self.encoding = tf.convert_to_tensor(self.encoding,dtype=tf.float32)
        super(PE_original,self).build(input_shape)


    def call(self, x):
        self.seq_len = x.shape[1]
        return self.encoding[:self.seq_len, :] + x


class PE_LSTM(Layer):
    """
    一个MxN的矩阵，可计算每个位置的位置情况
    """
    
    def __init__(self,emd_dim:int,seqL):
        self.emd_dim = emd_dim
        self.seqLL = seqL
        super(PE_LSTM,self).__init__()


    def build(self, input_shape):
        self.lstm = LSTM(self.emd_dim,activation='relu',return_sequences=True)
        super(PE_LSTM,self).build(input_shape)


    def call(self, x):
        lstm = self.lstm(x)
        return x + lstm
    
    # def compute_output_shape(self, input_shape):
    #     return (None,self.seqLL,self.emd_dim)
    



class MultiHeadAttention(Layer):
    def __init__(self, d_model,num_heads,seql,is_resnet=False,mask=None):
        self.is_resnet =is_resnet
        self.mask = mask
        self.num_heads = num_heads
        self.d_model = d_model
        self.seql = seql
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads
        super(MultiHeadAttention, self).__init__()

    def build(self, input_shape):
        self.WQ = Dense(self.d_model)
        self.WK = Dense(self.d_model)
        self.WV = Dense(self.d_model)
        self.dense = Dense(self.d_model)
        self.split = Reshape((self.seql,self.num_heads,self.depth))
        self.perpute = Permute([2, 1, 3])
        self.concat = Reshape((self.seql,self.d_model))
        super(MultiHeadAttention, self).build(input_shape)
    
    def split_heads(self, x):
        x = self.split(x)
        x = self.perpute(x)
        return x
    
    def call(self, x, mask=None):
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        # 缩放点积
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        dk = tf.cast(self.depth, tf.float32) # 缩放的维度
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if self.mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
        output = tf.matmul(attention_weights, v)
        res = tf.transpose(output, perm = [0, 2, 1, 3])
        out = self.concat(res)
        if self.is_resnet:
            return x + out
        else:
            return out
        



class EncoderLayer(Layer):
    def __init__(self, d_model,num_heads,seql,dropout_rate,mask=None):
        self.d_model = d_model
        self.num_head = num_heads
        self.seql = seql
        self.mask = mask
        self.dropout_rate = dropout_rate
        super(EncoderLayer, self).__init__()


    def build(self, input_shape):
        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)
        self.mha = MultiHeadAttention(d_model=self.d_model,num_heads=self.num_head,seql=self.seql,mask=self.mask)
        self.fnn = Dense(self.d_model,activation='relu')
        self.drop1 = Dropout(self.dropout_rate)
        self.drop2 = Dropout(self.dropout_rate)
        super(EncoderLayer, self).build(input_shape)
    

    
    def call(self,x,mask=None):
        attn = self.mha(x)
        attn = self.drop1(attn)
        n1 = self.layernorm1(x+attn)
        fnn = self.fnn(n1)
        fnn = self.drop2(fnn)
        n2 = self.layernorm2(fnn+n1)
        return n2
    


class Encoder(Layer):
    def __init__(self, d_model,num_heads,seql,dropout_rate,N_block,name=None):
        self.d_model = d_model
        self.num_head = num_heads
        self.seql = seql
        self.N = N_block
        self.dropout_rate = dropout_rate
        self.layers_list  = []
        super(Encoder, self).__init__(name=name)


    def build(self, input_shape):
        for _ in range(self.N):
            self.layers_list.append(EncoderLayer(d_model=self.d_model,num_heads=self.num_head,seql=self.seql,dropout_rate=self.dropout_rate))
        super(Encoder, self).build(input_shape)
    

    
    def call(self,x,mask=None):
        for mha_layer in self.layers_list:
            x = mha_layer(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape