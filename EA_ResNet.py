import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= IMPROVED EFFICIENT ATTENTION BLOCKS =============

class EfficientAttention(nn.Module):
    """
    Efficient Attention mechanism with linear complexity O(n)
    
    Formula: E(Q, K, V) = ρ_q(Q) (ρ_k(K)^T V)
    where ρ is the softmax function
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Apply softmax to Q and K separately (ρ_q and ρ_k)
        q_softmax = F.softmax(q, dim=-1)  # ρ_q(Q)
        k_softmax = F.softmax(k, dim=-2)  # ρ_k(K)
        
        # Efficient Attention: E(Q, K, V) = ρ_q(Q) (ρ_k(K)^T V)
        kv = torch.matmul(k_softmax.transpose(-2, -1), v)  # (B, num_heads, head_dim, head_dim)
        attn = torch.matmul(q_softmax, kv)  # (B, num_heads, N, head_dim)
        
        # Reshape and project
        attn = attn.transpose(0, 1).reshape(B, N, C)  # (B, N, C)
        x = self.proj(attn)
        x = self.proj_drop(x)
        
        return x


class AttentionBlock(nn.Module):
    """
    Attention Block with LayerNorm, Residual Connection, and MLP
    Improved integration for classification tasks
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP feedforward
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class EA_ResNet50(nn.Module):
    """
    Improved EA_ResNet50 with:
    - Multi-scale attention blocks (layer2 and layer3)
    - Improved attention with LayerNorm, Residual Connection, and MLP
    - Better integration with ResNet architecture for classification
    """
    def __init__(self, num_classes=1000, num_heads=8, attn_drop=0.1):
        super(EA_ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        
        # Layer2 with attention (512 channels = 128*4)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.attn_layer2 = AttentionBlock(dim=512, num_heads=num_heads, 
                                          attn_drop=attn_drop, drop=0.1)
        
        # Layer3 with attention (1024 channels = 256*4)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.attn_layer3 = AttentionBlock(dim=1024, num_heads=num_heads, 
                                          attn_drop=attn_drop, drop=0.1)
        
        # Layer4 without attention
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, 56, 56)
        
        x = self.layer1(x)  # (B, 256, 56, 56)
        
        # Layer2 + Attention
        x = self.layer2(x)  # (B, 512, 28, 28)
        B, C, H, W = x.shape
        x_attn = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_attn = self.attn_layer2(x_attn)
        x = x_attn.transpose(1, 2).reshape(B, C, H, W)  # (B, 512, 28, 28)
        
        # Layer3 + Attention
        x = self.layer3(x)  # (B, 1024, 14, 14)
        B, C, H, W = x.shape
        x_attn = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_attn = self.attn_layer3(x_attn)
        x = x_attn.transpose(1, 2).reshape(B, C, H, W)  # (B, 1024, 14, 14)
        
        x = self.layer4(x)  # (B, 2048, 7, 7)
        x = self.avgpool(x)  # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.fc(x)  # (B, num_classes)
        return x

# Create model
model = EA_ResNet50(num_classes=1000)
