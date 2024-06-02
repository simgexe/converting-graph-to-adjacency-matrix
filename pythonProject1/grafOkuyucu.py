import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def detect_nodes_and_edges(image_path):
    # Görüntüyü yükleme
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path)

    # Görüntüyü bulanıklaştırarak gürültüleri azaltma
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Kenarları algılamak için Canny kenar algılama
    edges = cv2.Canny(blurred, 50, 150)

    # Çemberleri tespit etme (düğümler)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=40)
    #Tespit edilen çemberleri node arrayine atma
    nodes = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        nodes = [(x, y) for x, y, r in circles]
        for (x, y, r) in circles:
            cv2.circle(img_color, (x, y), r, (0, 255, 0), 4)

    # Kenarları tespit etme
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=8)

    #Tespit edilen kenarları edges_list arrayine atma
    edges_list = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            edges_list.append(((x1, y1), (x2, y2)))

    # Düğümleri ve kenarları işaretlenmiş görüntüyü gösterme
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.show()

    return nodes, edges_list

    #Komşuluk matrisi oluşturma

def create_adjacency_matrix(nodes, edges):
    # Düğüm ve kenar verilerini alarak komşuluk matrisi oluşturur.
    n = len(nodes)
    # Düğüm sayısını belirler.
    adj_matrix = np.zeros((n, n), dtype=int)
    # n x n boyutlarında sıfır matrisi oluşturur.
    node_indices = {node: i for i, node in enumerate(nodes)}
    # Her düğümün indeksini belirleyen sözlük oluşturur.

    def find_closest_node(x, y):
        # Verilen koordinatlara en yakın düğümü bulur.
        closest_node = None
        min_dist = float('inf')

        for node in nodes:
            dist = (node[0] - x)**2 + (node[1] - y)**2
            # Öklidyen mesafenin karesini hesaplar.
            if dist < min_dist:
                min_dist = dist
                closest_node = node
                # Daha küçük mesafe bulunduğunda günceller.
        return closest_node
        # En yakın düğümü döndürür.

    for edge in edges:
        # Her kenar için döngü.
        node1 = find_closest_node(edge[0][0], edge[0][1])
        # Kenarın ilk ucuna en yakın düğümü bulur.
        node2 = find_closest_node(edge[1][0], edge[1][1])
        # Kenarın ikinci ucuna en yakın düğümü bulur.
        if node1 is not None and node2 is not None and node1 != node2:

            index1 = node_indices[node1]
            # İlk düğümün indeksini alır.
            index2 = node_indices[node2]
            # İkinci düğümün indeksini alır.
            adj_matrix[index1][index2] = 1
            adj_matrix[index2][index1] = 1
            # Matrisin karşılıklı elemanlarını 1 yapar (bağlantı belirtir).

    return adj_matrix
    # Oluşturulan komşuluk matrisini döndürür.

image_path = 'simple-graph-1.png'
nodes, edges = detect_nodes_and_edges(image_path)

# Düğümleri ve kenarları konsola yazdırma
print("Nodes:", nodes)
print("Edges:", edges)

adjacency_matrix = create_adjacency_matrix(nodes, edges)

# Komşuluk matrisini yazdırma
adjacency_df = pd.DataFrame(adjacency_matrix)
print(adjacency_df)
