class Spatial_Attention_layer(nn.Block):
    '''
    # 空间注意力层
    compute spatial attention scores 计算空间注意力得分
    '''
    def __init__(self, **kwargs):
        super(Spatial_Attention_layer, self).__init__(**kwargs)  # 采用 nn.Block 的初始化
        with self.name_scope():
            # 通过get函数从“共享”字典中检索，找不到则创建Parameter，声明需要名字和尺寸
            # 创建这些参数，并允许延迟初始化
            self.W_1 = self.params.get('W_1', allow_deferred_init=True)  # allow_deferred_init 允许延迟初始化
            self.W_2 = self.params.get('W_2', allow_deferred_init=True)
            self.W_3 = self.params.get('W_3', allow_deferred_init=True)
            self.b_s = self.params.get('b_s', allow_deferred_init=True)
            self.V_s = self.params.get('V_s', allow_deferred_init=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})  # 样本个数，顶点个数，特征个数，时间长度

        Returns
        ----------
        S_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, N, N)

        '''
        #get shape of input matrix x 获得输入矩阵的维数
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        #define the shape of params 延迟参数的形状
        self.W_1.shape = (num_of_timesteps,)  # W_1 \in R^{T_{r-1}}
        self.W_2.shape = (num_of_features, num_of_timesteps)  # ,W_2\ in R^{C_{r-1}\times T_{r-1}}
        self.W_3.shape = (num_of_features,)  # W_3 \in R^{C_{r-1}}
        self.b_s.shape = (1, num_of_vertices, num_of_vertices)  # b_s \in R^{1\times N \times N}
        self.V_s.shape = (num_of_vertices, num_of_vertices)  # V_s \in R^{        N \times N}
        for param in [self.W_1, self.W_2, self.W_3, self.b_s, self.V_s]:
            param._finish_deferred_init()  # 去完成原来要求延迟初始化的数据

        s = nd.dot(self.V_s.data(), nd.sigmoid(product + self.b_s.data()).transpose((1,2,0))).transpose(2,01)
        # V_s.shape = (N,N)
        #乘积+偏置送入sigmoid激活函数中， 激活后的结果进行转置 变为 (N,N,B)
        #v_s dot 加了偏置的结果 = (N,N) dot (N,N,B) = (N,N,B) 再转置
        # shape = (B,N,N)

        #标准化
        S = S - nd.max(S, axis = 1, keepdims = True)
        exp = nd.exp(S)
        S_normalized = exp / nd.sum(exp, axis=1, keepdims= True)
        return S_normalized

