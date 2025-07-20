import torch

patch_size = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]

if __name__ == '__main__':
    similarity = torch.zeros(16, len(patch_size)-1)
    for layer in range(0, 16):
        print(f'Layer {layer} Cosine Similarity:')
        
        for i in range(len(patch_size)-1):
            data1 = torch.load(f'/home/wyj24/project/VAR/feature_map_output/blocks.{layer}.ffn.fc2_{patch_size[i]**2}_output.pt')
            data2 = torch.load(f'/home/wyj24/project/VAR/feature_map_output/blocks.{layer}.ffn.fc2_{patch_size[i+1]**2}_output.pt')
            
            data1 = data1.view(patch_size[i], patch_size[i], 1024)
            data2 = data2.view(patch_size[i+1], patch_size[i+1], 1024)
            
            # interpolate data1 to match data2's shape [8, 8, 1024] -> [13, 13, 1024]
            data1_interp = torch.nn.functional.interpolate(
                data1.permute(2, 0, 1).unsqueeze(0),
                size=(patch_size[i+1], patch_size[i+1]),
                mode='bilinear',
                align_corners=True
            ).squeeze(0).permute(1, 2, 0)
            
            # data1_interp = data1_interp - data1_interp.mean()  # Normalize
            # data2 = data2 - data2.mean()
            
            cos_sim = torch.nn.functional.cosine_similarity(data1_interp.view(-1, 1024), data2.view(-1, 1024), dim=1)
            print(f'Cosine Similarity: {cos_sim.mean().item()}')
            similarity[layer, i] = cos_sim.mean().item()
        
        # plot the similarity matrix
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    for i in range(16):
        
        plt.plot(patch_size[:-1], similarity[i].numpy(), marker='o')
        plt.text(patch_size[-2], similarity[i, -1].item(), f'Layer {i}', fontsize=8, ha='right')
    plt.xticks(patch_size[:-1])
    plt.yticks(torch.arange(0, 1.1, 0.1))
    plt.title(f'Cosine Similarity')
    plt.xlabel('Patch Size')
    plt.ylabel('Cosine Similarity')
    plt.grid()
    plt.savefig(f'cosine_similarity_layer_d16_ffn_plot.png')
    plt.close()
    print('-' * 50)