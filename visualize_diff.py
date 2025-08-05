import torch

patch_size = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]

if __name__ == '__main__':
    for layer in range(0, 16):
        print(f'Layer {layer} Cosine Similarity:')
        similarity = torch.zeros(len(patch_size), len(patch_size))
        for i in range(len(patch_size)):
            for j in range(len(patch_size)):
                if j >= i:
                    data1 = torch.load(f'/home/wyj24/project/VAR/feature_map_output/blocks.{layer}.attn.proj_{patch_size[i]**2}_output.pt')
                    data2 = torch.load(f'/home/wyj24/project/VAR/feature_map_output/blocks.{layer}.attn.proj_{patch_size[j]**2}_output.pt')
                    
                    data1 = data1.view(data1.shape[0], patch_size[i], patch_size[i], 1024)
                    data2 = data2.view(data2.shape[0], patch_size[j], patch_size[j], 1024)
                    
                    # interpolate data1 to match data2's shape [8, 8, 1024] -> [13, 13, 1024]
                    data1_interp = torch.nn.functional.interpolate(
                        data1.permute(0, 3, 1, 2),
                        size=(patch_size[j], patch_size[j]),
                        mode='bilinear',
                        align_corners=True
                    ).permute(0, 2, 3, 1)
                    data1_interp = data1_interp.view(data1_interp.shape[0], -1, 1024)
                    
                    # data1_interp = data1_interp - data1_interp.mean()  # Normalize
                    # data2 = data2 - data2.mean()
                    
                    cos_sim = torch.nn.functional.cosine_similarity(data1_interp.view(-1, 1024), data2.view(-1, 1024), dim=1)
                    print(f'Cosine Similarity: {cos_sim.mean().item()}')
                    similarity[i, j] = cos_sim.mean().item()
        
        # plot the similarity matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity, annot=True, fmt=".2f", cmap='coolwarm',
                    xticklabels=patch_size, yticklabels=patch_size)
        plt.title(f'Cosine Similarity Matrix for Layer {layer}')
        plt.xlabel('Patch Size')
        plt.ylabel('Patch Size')
        plt.savefig(f'cosine_similarity/cosine_similarity_layer_{layer}_d16_ffn.png')
        plt.close()
        print(f'Cosine similarity matrix for layer {layer} saved as cosine_similarity_layer_{layer}_d16_ffn.png')
        print('-' * 50)