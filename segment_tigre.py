import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

caminho = "pdi_tf/tigre01.png"  
imagem = cv2.imread(caminho)
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Transformar a imagem em uma matriz de pixels (para aplicar o k-means)
pixels = imagem_rgb.reshape((-1, 3))

# Aplicar k-means 
k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

labels = kmeans.labels_
clusters = kmeans.cluster_centers_

# Selecionar a máscara do cluster 4 (foi a que melhor segmentou)
mascara_c4 = (labels == 4)


imagem_c4 = np.zeros_like(pixels)
imagem_c4[mascara_c4] = pixels[mascara_c4]
imagem_c4 = imagem_c4.reshape(imagem_rgb.shape)



# Carregar a ground truth com canal alfa
gt_img = Image.open("pdi_tf/tigre01_gt.png")
gt_alpha = np.array(gt_img.getchannel("A"))  # Canal alfa da ground truth

# Criar a máscara binária para o ground truth (1 para o objeto, 0 para o fundo)
gt_mask = gt_alpha > 128  # Threshold de binarização

# Criar a máscara binária para o cluster segmentado
seg_mask = mascara_c4.reshape(imagem_rgb.shape[:2])  # Máscara segmentada (reshape para 2D)

# Calcular o Dice Score
intersection = np.logical_and(seg_mask, gt_mask).sum()
seg_sum = seg_mask.sum()
gt_sum = gt_mask.sum()
dice_score = (2 * intersection) / (seg_sum + gt_sum)

print(f"Dice Score: {dice_score:.4f}")



plt.imshow(imagem_c4)
plt.axis("off")
plt.title("Cluster 4")
plt.show()
