import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Suponha que temos uma classe Dataset para YOLO
class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))

        # Carregar imagem
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)

        # Redimensionar imagem
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Carregar anotações
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
                        # Converter para coordenadas da caixa [x1, y1, x2, y2] normalizadas
                        x1 = (x_center - width/2) * self.img_size
                        y1 = (y_center - height/2) * self.img_size
                        x2 = (x_center + width/2) * self.img_size
                        y2 = (y_center + height/2) * self.img_size
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        # Suponha que temos uma transformação para o modelo
        if self.transform:
            image = self.transform(image)

        return image, boxes, labels

def load_test_data_yolo(dataset_path, client_id, batch_size=16, img_size=640):
    """
    Carrega o dataset de teste para o cliente `client_id` no formato YOLO.

    Args:
        dataset_path (str): Caminho base do dataset (ex: "COCO128/")
        client_id (int): ID do cliente
        batch_size (int): Tamanho do lote
        img_size (int): Tamanho da imagem para redimensionamento

    Returns:
        DataLoader: DataLoader para o dataset de teste do cliente
    """
    test_path = os.path.join(dataset_path, "test")
    client_images_dir = os.path.join(test_path, "images", f"client_{client_id:02d}")
    client_labels_dir = os.path.join(test_path, "labels", f"client_{client_id:02d}")

    # Verificar se o diretório existe
    if not os.path.exists(client_images_dir):
        raise FileNotFoundError(f"Diretório de imagens do cliente {client_id} não encontrado: {client_images_dir}")

    dataset = YOLODataset(client_images_dir, client_labels_dir, img_size=img_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return data_loader

def collate_fn(batch):
    """
    Função para collate dos dados em batch, pois as imagens podem ter números diferentes de objetos.
    """
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, boxes, labels

def evaluate_model_yolo(model, test_loader, device):
    """
    Avalia o modelo YOLO no test_loader.

    Args:
        model: Modelo YOLO
        test_loader: DataLoader de teste
        device: Dispositivo (ex: 'cuda' ou 'cpu')

    Returns:
        tuple: (mAP, avg_loss)
    """
    model.eval()
    total_loss = 0
    total_batches = 0

    # Listas para calcular mAP
    pred_boxes = []
    pred_scores = []
    pred_labels = []
    true_boxes = []
    true_labels = []

    # Suponha que o modelo YOLO retorna (loss, detections) quando em modo de treino e apenas detections quando em modo de eval?
    # Vamos assumir que temos um modelo que retorna a perda quando passamos as anotações, e detecções quando não.

    with torch.no_grad():
        for images, target_boxes, target_labels in test_loader:
            images = images.to(device)
            # Converter as anotações para o formato esperado pelo modelo (depende do modelo)
            # Aqui, vamos assumir que o modelo espera uma lista de dicionários com 'boxes' e 'labels'
            targets = []
            for boxes, labels in zip(target_boxes, target_labels):
                targets.append({'boxes': boxes.to(device), 'labels': labels.to(device)})

            # Forward pass: assumindo que o modelo retorna a perda e as detecções
            # Se o modelo tiver um método `evaluate` que retorna a perda, use-o.
            # Caso contrário, podemos ter que calcular a perda manualmente.

            # Vamos assumir que o modelo retorna um dicionário com 'loss' e 'detections'
            outputs = model(images, targets)  # Isso é apenas um exemplo, a implementação real pode variar

            # Suponha que a saída inclui a perda
            loss = outputs['loss']
            total_loss += loss.item()
            total_batches += 1

            # Para mAP, precisamos das detecções e das anotações verdadeiras
            # Vamos assumir que as detecções estão em outputs['detections']
            # Formato das detecções: [batch_size, num_detections, 6] (x1, y1, x2, y2, score, class)
            detections = outputs['detections']

            # Armazenar para calcular mAP
            for i in range(len(detections)):
                pred_boxes.append(detections[i][:, :4].cpu())
                pred_scores.append(detections[i][:, 4].cpu())
                pred_labels.append(detections[i][:, 5].cpu())
                true_boxes.append(target_boxes[i])
                true_labels.append(target_labels[i])

    avg_loss = total_loss / total_batches

    # Calcular mAP
    # Vamos usar uma função de cálculo de mAP (ex: from torchmetrics.detection import MeanAveragePrecision)
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(pred_boxes, pred_labels, true_boxes, true_labels)
    result = metric.compute()
    map_value = result['map'].item()  # mAP

    return map_value, avg_loss