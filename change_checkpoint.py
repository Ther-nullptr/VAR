import torch
from collections import defaultdict

def merge_qkv_bias(checkpoint):
    # 创建一个字典来分组相同block的q/v/zero_k偏置
    bias_groups = defaultdict(dict)
    
    # 遍历所有键并分组
    for key in list(checkpoint.keys()):
        parts = key.split('.')
        if len(parts) >= 4 and parts[-1] in ['q_bias', 'v_bias', 'zero_k_bias']:
            # 提取block编号和偏置类型
            block_num = parts[1]
            bias_type = parts[-1]
            
            # 添加到分组字典
            bias_groups[block_num][bias_type] = checkpoint[key]
            # 删除原键
            del checkpoint[key]
    
    # 合并每组偏置
    print(f'Found {len(bias_groups)} blocks with biases.')
    for block_num, biases in bias_groups.items():
        # 确保三种偏置都存在
        if 'q_bias' in biases and 'v_bias' in biases and 'zero_k_bias' in biases:
            # 按q, zero_k, v的顺序拼接
            merged_bias = torch.cat([biases['q_bias'], biases['zero_k_bias'], biases['v_bias']])
            new_key = f'blocks.{block_num}.attn.mat_qkv.bias'
            checkpoint[new_key] = merged_bias
    
    return checkpoint

# 使用示例
if __name__ == "__main__":
    # 加载你的checkpoint
    checkpoint = torch.load('/home/wyj24/models/VAR/var_d20.pth')
    
    # 合并偏置
    merged_checkpoint = merge_qkv_bias(checkpoint)
    
    # 保存新的checkpoint
    print(f'Merged checkpoint keys: {list(merged_checkpoint.keys())}')
    torch.save(merged_checkpoint, '/home/wyj24/models/VAR/var_d20_new.pth')