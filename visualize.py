import torch

data1 = torch.load('/home/wyj24/project/VAR/example0.pt')
data2 = torch.load('/home/wyj24/project/VAR/example1.pt')

# visualize the data1 and data2 in colored images
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(data1.cpu().numpy(), cmap='coolwarm')
plt.title('Data1')
plt.subplot(1, 2, 2)
plt.imshow(data2.cpu().numpy(), cmap='coolwarm')
plt.title('Data2')
plt.legend(['Data1', 'Data2'])
plt.tight_layout()
plt.savefig('data_visualization.png')


data1 = data1.view(10, 10, 1024)
data2 = data2.view(13, 13, 1024)
data1_interp = torch.nn.functional.interpolate(
    data1.permute(2, 0, 1).unsqueeze(0),
    size=(13, 13),
    mode='bilinear',
    align_corners=True
).squeeze(0).permute(1, 2, 0)
cos_sim = torch.nn.functional.cosine_similarity(data1_interp.view(-1, 1024), data2.view(-1, 1024), dim=1)
print(f'Cosine Similarity: {cos_sim.mean().item()}')