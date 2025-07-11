import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer


# 配置参数
class Config:
    def __init__(self):
        # MRI参数
        self.mri_size = (192, 256, 256)  # 输入尺寸(D,H,W)
        self.patch_size = 16  # 16mm³的物理patch大小
        self.in_channels = 1

        # 文本参数
        self.text_model = 'emilyalsentzer/Bio_ClinicalBERT'
        self.max_text_len = 128

        # 模型参数
        self.embed_dim = 768
        self.num_heads = 8
        self.depth = 6
        self.mlp_ratio = 4
        self.drop_rate = 0.2
        self.num_classes = 3

        # 初始化验证
        self._validate()

    def _validate(self):
        assert all(s % self.patch_size == 0 for s in self.mri_size), "所有维度必须能被patch_size整除"
        print("配置验证通过！")


config = Config()


# 1. 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.text_model)
        self.proj = nn.Linear(768, config.embed_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        # 冻结大部分层
        for param in self.bert.parameters():
            param.requires_grad = False
        # 只微调最后两层
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.dropout(self.proj(cls_token))


# 2. MRI编码器
class MRIPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid_size = (
            config.mri_size[0] // config.patch_size,  # 12
            config.mri_size[1] // config.patch_size,  # 16
            config.mri_size[2] // config.patch_size  # 16
        )
        self.num_patches = np.prod(self.grid_size)

        # 3D卷积划分patch,(B,1,12,16,16)<-(B,768,12,16,16)
        self.proj = nn.Conv3d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, C, D, H, W] -> [B, embed_dim, 12, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 3072, embed_dim]
        return self.norm(x)


# 3. Transformer模块
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            config.embed_dim,
            config.num_heads,
            dropout=config.drop_rate
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.drop_rate),
            nn.Linear(config.embed_dim * config.mlp_ratio, config.embed_dim),
            nn.Dropout(config.drop_rate)
        )

    def forward(self, x):
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x):
        x = self.attn(x, x, x, need_weights=False)[0]
        return x

    def _ff_block(self, x):
        return self.mlp(x)


# 4. 完整多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.mri_encoder = MRIPatchEmbed(config)

        # 计算MRI patch数量
        grid_size = (
            config.mri_size[0] // config.patch_size,
            config.mri_size[1] // config.patch_size,
            config.mri_size[2] // config.patch_size
        )
        num_patches = np.prod(grid_size)

        # 图像主导的设计
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # 添加 text_token 与 MRI 的各个 patch 进行更有效的交互，这里 B 设为 1，PyTorch 会自动将其广播到与输入批次相同的大小
        self.text_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # 位置编码的设计是每个位置有一个固定的向量，然后应用到所有批次样本上：[1, 序列长度, embed_dim] 广播到批次维度 [B, 序列长度, embed_dim]
        self.text_pos_embed = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) 
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, config.embed_dim)) # +2表示与CLS和文本拼接
        self.pos_drop = nn.Dropout(config.drop_rate)

        # Transformer
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.text_token, std=0.005)
        nn.init.trunc_normal_(self.text_pos_embed, std=0.005)

    def forward(self, mri, text_ids, text_mask):
        B = mri.shape[0]
        mri_feat = self.mri_encoder(mri)
        text_feat = self.text_encoder(text_ids, text_mask).unsqueeze(1)
        text_feat = text_feat + self.text_token + self.text_pos_embed

        # 拼接顺序：[图像cls_token, 图像特征, 文本特征]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, mri_feat, text_feat), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x[:, 0])  # 仅使用图像cls_token

        return self.head(x)


# 随机生成MRI测试
if __name__ == "__main__":
    # 初始化模型、分词器、损失函数、优化器
    model = MultimodalModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # ----------------------
    # 1. 准备模拟训练数据（实际中需替换为真实数据集）
    # ----------------------
    # 模拟100个样本的训练数据（批次大小=2，共50个批次）
    num_samples = 100
    batch_size = 2
    num_batches = num_samples // batch_size


    # 生成模拟数据（MRI+文本+标签）
    def generate_batch(batch_idx):
        # 随机生成MRI数据（[B, C, D, H, W]）
        mri = torch.randn(batch_size, 1, 192, 256, 256)
        # 随机生成2条文本（模拟不同样本）
        texts = [
            f"patient case {batch_idx * 2 + i}: symptoms include headache"
            for i in range(batch_size)
        ]
        # 分词器处理文本
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        # 随机生成标签（0/1/2三类）
        labels = torch.randint(0, config.num_classes, (batch_size,))
        return mri, inputs["input_ids"], inputs["attention_mask"], labels


    # ----------------------
    # 2. 训练循环（核心部分）
    # ----------------------
    num_epochs = 3  # 训练轮数（根据实际情况调整，通常需要10~100轮）

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式（启用dropout等训练特有的层）
        total_loss = 0.0  # 记录当前轮次的总损失

        # 按批次迭代训练数据
        for batch_idx in range(num_batches):
            # 获取当前批次数据
            mri, text_ids, text_mask, labels = generate_batch(batch_idx)

            # 清空梯度（避免上一轮梯度累积）
            optimizer.zero_grad()

            # 前向传播：计算模型输出
            outputs = model(mri, text_ids, text_mask)  # 形状：[batch_size, 3]

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()  # 累加损失

            # 反向传播：计算梯度
            loss.backward()

            # 参数更新：根据梯度调整模型参数
            optimizer.step()

            # 打印批次日志（每10个批次打印一次）
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{num_batches}], Loss: {loss.item():.4f}")

        # 计算当前轮次的平均损失
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}] 完成！平均损失: {avg_loss:.4f}\n")

    print("训练结束！")
