import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

caminho = "pdi_tf/arara_azul.png"  
imagem = cv2.imread(caminho)
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Transformar a imagem em uma matriz de pixels (para aplicar o k-means)
pixels = imagem_rgb.reshape((-1, 3))

# Aplicar k-means 
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

labels = kmeans.labels_
clusters = kmeans.cluster_centers_

# Selecionar a máscara do cluster 4 (foi a que melhor segmentou)
mascara_c4 = (labels == 1)

imagem_c4 = np.zeros_like(pixels)
imagem_c4[mascara_c4] = pixels[mascara_c4]
imagem_c4 = imagem_c4.reshape(imagem_rgb.shape)




plt.imshow(imagem_c4)
plt.axis("off")
plt.title("Cluster 4")
plt.show()
