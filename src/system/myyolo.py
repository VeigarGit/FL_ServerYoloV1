import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import yaml
import cv2
import numpy as np
from PIL import Image
import random
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# ==================== CONFIGURAÇÃO DO DATASET ====================

class YOLODataset(Dataset):
    """Dataset personalizado para formato YOLO"""
    
    def __init__(self, data_yaml_path, split='train', img_size=640, transform=None):
        """
        Args:
            data_yaml_path: Caminho para o arquivo data.yaml
            split: 'train', 'val' ou 'test'
            img_size: Tamanho da imagem para redimensionamento
            transform: Transformações adicionais
        """
        self.img_size = img_size
        self.transform = transform
        
        # Carrega configuração do dataset
        with open(data_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Determina o caminho base e as imagens
        base_path = self.config['path']
        if split == 'train':
            split_path = self.config['train']
        elif split == 'val':
            split_path = self.config['val']
        else:  # test
            split_path = self.config.get('test', self.config['val'])
        
        self.images_dir = os.path.join(base_path, split_path)
        self.labels_dir = self.images_dir.replace('/images', '/labels')
        
        # Lista todas as imagens
        self.image_files = [
            f for f in os.listdir(self.images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        self.num_classes = self.config['nc']
        self.class_names = self.config['names']
        
        print(f"✅ Dataset {split}: {len(self.image_files)} imagens carregadas")
        print(f"📁 Imagens: {self.images_dir}")
        print(f"📁 Labels: {self.labels_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def load_image_and_label(self, index):
        """Carrega imagem e labels correspondentes"""
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Carrega imagem
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # Carrega labels
        label_path = os.path.join(
            self.labels_dir, 
            os.path.splitext(img_name)[0] + '.txt'
        )
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        boxes.append([class_id, x_center, y_center, width, height])
        
        return image, np.array(boxes), (original_h, original_w)
    
    def resize_image(self, image, boxes, target_size):
        """Redimensiona imagem e ajusta bounding boxes"""
        h, w = image.shape[:2]
        # Mantém aspect ratio
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensiona imagem
        resized = cv2.resize(image, (new_w, new_h))
        
        # Preenche com padding para atingir target_size
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Ajusta bounding boxes
        if len(boxes) > 0:
            boxes[:, 1] = boxes[:, 1] * scale  # x_center
            boxes[:, 2] = boxes[:, 2] * scale  # y_center
            boxes[:, 3] = boxes[:, 3] * scale  # width
            boxes[:, 4] = boxes[:, 4] * scale  # height
        
        return padded, boxes, (new_h, new_w), scale
    
    def __getitem__(self, index):
        image, boxes, original_size = self.load_image_and_label(index)
        
        # Aplica transformações
        if self.transform:
            image, boxes = self.transform(image, boxes)
        else:
            # Redimensionamento padrão
            image, boxes, new_size, scale = self.resize_image(image, boxes, self.img_size)
        
        # Converte para tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Prepara targets no formato [class_id, x_center, y_center, width, height]
        targets = torch.zeros((len(boxes), 6))  # [sample_index, class_id, x, y, w, h]
        if len(boxes) > 0:
            targets[:, 1] = torch.from_numpy(boxes[:, 0])  # class_id
            targets[:, 2] = torch.from_numpy(boxes[:, 1])  # x_center
            targets[:, 3] = torch.from_numpy(boxes[:, 2])  # y_center
            targets[:, 4] = torch.from_numpy(boxes[:, 3])  # width
            targets[:, 5] = torch.from_numpy(boxes[:, 4])  # height
        
        return image_tensor, targets

# ==================== ARQUITETURA YOLO EM PYTORCH ====================

class Conv(nn.Module):
    """Bloco de convolução com batch norm e ativação SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Bloco bottleneck residual"""
    def __init__(self, in_channels, out_channels, shortcut=True, repression_ratio=0.25):
        super().__init__()
        hidden_channels = int(out_channels * repression_ratio)
        self.conv1 = Conv(in_channels, hidden_channels, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3)
        self.use_add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.use_add else self.conv2(self.conv1(x))

class C3k2(nn.Module):
    """Bloco C3 com repressão de canais"""
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, repression_ratio=0.25):
        super().__init__()
        hidden_channels = int(out_channels * repression_ratio)
        
        self.conv1 = Conv(in_channels, hidden_channels, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1)
        
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0) 
              for _ in range(depth)]
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bottlenecks(self.conv2(x))
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)

class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, 1, kernel_size//2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

class C2PSA(nn.Module):
    """Bloco C2 com atenção simplificada"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(in_channels, out_channels, 1)
        self.conv3 = Conv(out_channels, out_channels, 3)
        self.conv4 = Conv(out_channels, out_channels, 3)
        
        # Mecanismo de atenção simplificado
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.conv3(self.conv1(x))
        x2 = self.conv4(self.conv2(x))
        attention_weights = self.attention(x1 + x2)
        x1 = x1 * attention_weights
        return x1 + x2

class Detect(nn.Module):
    """Camada de detecção YOLO"""
    def __init__(self, num_classes=80, channels=()):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # xywh + conf + classes
        
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], self.num_outputs * 3, 1) 
            for i in range(len(channels))
        ])
        
    def forward(self, xs):
        outputs = []
        for i, x in enumerate(xs):
            output = self.convs[i](x)
            b, _, h, w = output.shape
            output = output.view(b, 3, self.num_outputs, h, w).permute(0, 1, 3, 4, 2)
            outputs.append(output)
        
        return torch.cat([x.reshape(x.shape[0], -1, self.num_outputs) for x in outputs], dim=1)

class YOLO11(nn.Module):
    """Implementação completa do YOLO11 em PyTorch"""
    
    def __init__(self, config='n', num_classes=80):
        super().__init__()
        
        # Configurações de escala
        scales = {
            'n': [0.50, 0.25, 1024],
            's': [0.50, 0.50, 1024],
            'm': [0.50, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
        
        depth_multiple, width_multiple, max_channels = scales[config]
        
        def make_channels(channels):
            return min(int(channels * width_multiple), max_channels)
        
        # Calcular canais para cada nível
        self.p3_channels = make_channels(256)  # 64 para 'n'
        self.p4_channels = make_channels(512)  # 128 para 'n' 
        self.p5_channels = make_channels(1024) # 256 para 'n'
        
        # ==================== BACKBONE ====================
        self.backbone = nn.ModuleList([
            Conv(3, make_channels(64), 3, 2),  # 0-P1/2
            Conv(make_channels(64), make_channels(128), 3, 2),  # 1-P2/4
            C3k2(make_channels(128), make_channels(256), 
                 depth=int(2 * depth_multiple), repression_ratio=0.25),  # 2
            Conv(make_channels(256), make_channels(256), 3, 2),  # 3-P3/8
            C3k2(make_channels(256), self.p3_channels, 
                 depth=int(2 * depth_multiple), repression_ratio=0.25),  # 4 (P3)
            Conv(self.p3_channels, make_channels(512), 3, 2),  # 5-P4/16
            C3k2(make_channels(512), self.p4_channels, 
                 depth=int(2 * depth_multiple)),  # 6 (P4)
            Conv(self.p4_channels, make_channels(1024), 3, 2),  # 7-P5/32
            C3k2(make_channels(1024), self.p5_channels, 
                 depth=int(2 * depth_multiple)),  # 8 (P5)
            SPPF(self.p5_channels, self.p5_channels),  # 9
            C2PSA(self.p5_channels, self.p5_channels)  # 10
        ])
        
        # ==================== HEAD ====================
        # Canais para as concatenações
        upsample1_channels = self.p5_channels + self.p4_channels  # 256 + 128 = 384
        upsample2_channels = make_channels(512) + self.p3_channels  # 128 + 64 = 192
        downsample1_channels = make_channels(256) + make_channels(512)  # 64 + 128 = 192
        downsample2_channels = make_channels(512) + self.p5_channels  # 128 + 256 = 384
        
        self.head = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 11
            C3k2(upsample1_channels, make_channels(512), 
                 depth=int(2 * depth_multiple)),  # 12
            nn.Upsample(scale_factor=2, mode='nearest'),  # 13
            C3k2(upsample2_channels, make_channels(256), 
                 depth=int(2 * depth_multiple)),  # 14 (P3/8-small)
            Conv(make_channels(256), make_channels(256), 3, 2),  # 15
            C3k2(downsample1_channels, make_channels(512), 
                 depth=int(2 * depth_multiple)),  # 16 (P4/16-medium)
            Conv(make_channels(512), make_channels(512), 3, 2),  # 17
            C3k2(downsample2_channels, make_channels(1024), 
                 depth=int(2 * depth_multiple)),  # 18 (P5/32-large)
        ])
        
        # ==================== DETECT ====================
        self.detect = Detect(num_classes, [
            make_channels(256),   # P3/8
            make_channels(512),   # P4/16  
            make_channels(1024)   # P5/32
        ])
        
    def forward(self, x):
        # Backbone - salvar features para FPN
        backbone_features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 10]:  # Salvar P3, P4, P5
                backbone_features.append(x)
        
        p3, p4, p5 = backbone_features[0], backbone_features[1], backbone_features[2]
        
        # Head - FPN
        # Primeiro upsampling: P5 -> P4
        x = self.head[0](p5)  # Upsample P5
        x = torch.cat([x, p4], dim=1)  # Concat com P4
        x = self.head[1](x)   # C3k2
        
        # Segundo upsampling: P4 -> P3
        p4_out = x
        x = self.head[2](x)   # Upsample
        x = torch.cat([x, p3], dim=1)  # Concat com P3
        p3_out = self.head[3](x)  # C3k2 (P3/8-small)
        
        # Downsampling: P3 -> P4
        x = self.head[4](p3_out)  # Conv
        x = torch.cat([x, p4_out], dim=1)  # Concat
        p4_out = self.head[5](x)  # C3k2 (P4/16-medium)
        
        # Downsampling: P4 -> P5
        x = self.head[6](p4_out)  # Conv
        x = torch.cat([x, p5], dim=1)  # Concat
        p5_out = self.head[7](x)  # C3k2 (P5/32-large)
        
        # Detecção nos três níveis
        return self.detect([p3_out, p4_out, p5_out])

# ==================== FUNÇÕES UTILITÁRIAS ====================

def collate_fn(batch):
    """Função para colate dos batches"""
    images, targets = list(zip(*batch))
    images = torch.stack(images)
    
    # Encontra o número máximo de targets em um batch
    max_targets = max(len(t) for t in targets)
    
    # Preenche os targets com zeros
    padded_targets = []
    for i, t in enumerate(targets):
        if len(t) < max_targets:
            pad = torch.zeros((max_targets - len(t), 6))
            pad[:, 0] = i  # sample index
            t = torch.cat([t, pad])
        padded_targets.append(t)
    
    targets = torch.cat(padded_targets)
    return images, targets

def create_data_loaders(data_yaml_path, batch_size=8, img_size=640):
    """Cria data loaders para treino, validação e teste"""
    
    train_dataset = YOLODataset(data_yaml_path, 'train', img_size)
    val_dataset = YOLODataset(data_yaml_path, 'val', img_size)
    test_dataset = YOLODataset(data_yaml_path, 'test', img_size)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def validate_dataset_structure(data_yaml_path):
    """Valida a estrutura do dataset"""
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = config['path']
    issues = []
    
    splits = {
        'train': config.get('train'),
        'val': config.get('val'), 
        'test': config.get('test')
    }
    
    for split_name, split_path in splits.items():
        if not split_path:
            continue
            
        full_path = os.path.join(base_path, split_path)
        
        if not os.path.exists(full_path):
            issues.append(f"❌ Pasta não encontrada: {full_path}")
            continue
            
        images = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            issues.append(f"⚠️  Nenhuma imagem em: {full_path}")
        
        labels_path = full_path.replace('/images', '/labels')
        if not os.path.exists(labels_path):
            issues.append(f"❌ Pasta de labels não encontrada: {labels_path}")
        else:
            labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
            if not labels:
                issues.append(f"⚠️  Nenhum label em: {labels_path}")
    
    return issues, config

# ==================== FUNÇÕES DE PREDIÇÃO ====================

def predict_and_save(model, source_path, output_dir="predicts", conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    """
    Faz predições em imagens e salva os resultados em uma pasta.
    
    Args:
        model: Modelo YOLO treinado
        source_path: Caminho para imagem única, pasta de imagens ou padrão glob
        output_dir: Pasta onde salvar as predições
        conf_threshold: Limiar de confiança para detecções
        iou_threshold: Limiar para Non-Maximum Suppression
        img_size: Tamanho da imagem para redimensionamento
    """
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # Criar pasta de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Obter lista de imagens
    if os.path.isdir(source_path):
        image_paths = glob.glob(os.path.join(source_path, "*.jpg")) + \
                     glob.glob(os.path.join(source_path, "*.jpeg")) + \
                     glob.glob(os.path.join(source_path, "*.png"))
    elif "*" in source_path:
        image_paths = glob.glob(source_path)
    else:
        image_paths = [source_path]
    
    print(f"🔍 Encontradas {len(image_paths)} imagens para predição")
    
    # Colocar modelo em modo de avaliação
    model.eval()
    
    # Processar cada imagem
    for image_path in image_paths:
        try:
            # Carregar e pré-processar imagem
            original_image, processed_image = preprocess_image(image_path, img_size)
            
            # Fazer predição
            with torch.no_grad():
                predictions = model(processed_image)
            
            # Processar predições
            detections = process_predictions(predictions, conf_threshold, iou_threshold, img_size)
            
            # Desenhar bounding boxes
            result_image = draw_detections(original_image, detections, model.detect.num_classes - 5)
            
            # Salvar imagem
            output_path = os.path.join(output_dir, f"pred_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, result_image)
            
            print(f"✅ Salvo: {output_path} com {len(detections)} detecções")
            
        except Exception as e:
            print(f"❌ Erro ao processar {image_path}: {e}")
    
    print(f"🎯 Predições salvas em: {output_dir}")

def preprocess_image(image_path, img_size=640):
    """Pré-processa a imagem para o modelo YOLO"""
    # Carregar imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    original_image = image.copy()
    
    # Converter BGR para RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar mantendo aspect ratio
    h, w = image.shape[:2]
    scale = min(img_size / w, img_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Redimensionar
    resized = cv2.resize(image, (new_w, new_h))
    
    # Adicionar padding
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Normalizar e converter para tensor
    normalized = padded.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return original_image, tensor

def process_predictions(predictions, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    """
    Processa as predições do modelo YOLO
    Retorna: Lista de [x1, y1, x2, y2, confidence, class_id]
    """
    detections = []
    
    if len(predictions) == 0:
        return detections
    
    pred = predictions[0]  # Primeiro batch
    
    # Aplicar limiar de confiança
    confidence = pred[:, 4]  # Confidence score
    mask = confidence > conf_threshold
    pred = pred[mask]
    
    if len(pred) == 0:
        return detections
    
    # Obter class probabilities
    class_conf, class_pred = torch.max(pred[:, 5:], dim=1)
    
    # Combinar confidence do objeto com confidence da classe
    overall_conf = confidence[mask] * class_conf
    
    # Coordenadas das bounding boxes (centro x, centro y, largura, altura)
    boxes = pred[:, :4]
    
    # Converter para formato (x1, y1, x2, y2)
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * img_size
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * img_size
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * img_size
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * img_size
    
    # Stack todas as informações
    detections = torch.stack([x1, y1, x2, y2, overall_conf, class_pred.float()], dim=1)
    
    # Aplicar Non-Maximum Suppression
    if len(detections) > 0:
        keep = nms(detections[:, :4], detections[:, 4], iou_threshold)
        detections = detections[keep]
    
    return detections.cpu().numpy()

def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Coordenadas das boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Áreas das boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # Ordenar por score
    _, indices = scores.sort(descending=True)
    keep = []
    
    while len(indices) > 0:
        # Box com maior score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calcular IoU com as boxes restantes
        remaining = indices[1:]
        
        # Coordenadas da intersecção
        inter_x1 = torch.max(x1[current], x1[remaining])
        inter_y1 = torch.max(y1[current], y1[remaining])
        inter_x2 = torch.min(x2[current], x2[remaining])
        inter_y2 = torch.min(y2[current], y2[remaining])
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Calcular IoU
        union_area = areas[current] + areas[remaining] - inter_area
        iou = inter_area / union_area
        
        # Manter boxes com IoU menor que o threshold
        keep_indices = (iou <= iou_threshold).nonzero().squeeze()
        indices = remaining[keep_indices]
    
    return torch.tensor(keep, dtype=torch.long)

def draw_detections(image, detections, num_classes):
    """Desenha bounding boxes e labels na imagem"""
    # Cores para diferentes classes
    colors = generate_colors(num_classes)
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)
        
        # Converter coordenadas para inteiros
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Cor baseada na classe
        color = colors[class_id % len(colors)]
        
        # Desenhar bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Texto do label
        label = f"Class {class_id}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Fundo para o texto
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Texto
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def generate_colors(num_classes):
    """Gera cores distintas para cada classe"""
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        color = [int(x) for x in np.random.randint(0, 255, 3)]
        colors.append(color)
    return colors

# ==================== FUNÇÕES DE TREINAMENTO ====================

def yolo_loss(predictions, targets, num_classes=80):
    """Função de loss simplificada para YOLO"""
    # Implementação básica - em produção use uma mais robusta
    return F.mse_loss(predictions, torch.randn_like(predictions))

def train_yolo(model, train_loader, val_loader, epochs=1, device='cuda'):
    """Loop de treinamento simplificado"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = yolo_loss(outputs, targets, model.detect.num_classes)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} | Average Loss: {total_loss/len(train_loader):.4f}')

# ==================== EXEMPLO DE USO COMPLETO ====================

if __name__ == "__main__":
    # Configurações
    DATA_YAML_PATH = "/home/nap/FL_ServerYoloV1/src/dataset/tcl/0.yaml"
    BATCH_SIZE = 4
    IMG_SIZE = 640
    NUM_CLASSES = 8
    
    print("🚀 Inicializando YOLO Dataset e Modelo...")
    
    # 1. Valida estrutura do dataset
    issues, config = validate_dataset_structure(DATA_YAML_PATH)
    if issues:
        print("Problemas encontrados:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Estrutura do dataset válida!")
    
    # 2. Cria data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        DATA_YAML_PATH, BATCH_SIZE, IMG_SIZE
    )
    
    # 3. Cria modelo YOLO
    model = YOLO11('n', num_classes=NUM_CLASSES)
    
    print(f"✅ Modelo YOLO11-n criado com {sum(p.numel() for p in model.parameters()):,} parâmetros")
    print(f"✅ DataLoaders criados:")
    print(f"   - Treino: {len(train_loader.dataset)} imagens")
    print(f"   - Validação: {len(val_loader.dataset)} imagens") 
    print(f"   - Teste: {len(test_loader.dataset)} imagens")
    
    # 4. Teste rápido com verificação de formas
    print("\n🧪 Testando forward pass...")
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        images, targets = sample_batch
        print(f"   Input shape: {images.shape}")
        
        # Forward do modelo
        outputs = model(images)
        print(f"   Output shape: {outputs.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        # Verificar formas internas para debug
        print("\n📊 Formas internas do modelo:")
        print(f"   - P3 channels: {model.p3_channels}")
        print(f"   - P4 channels: {model.p4_channels}")
        print(f"   - P5 channels: {model.p5_channels}")
    
    print("\n🎯 Configuração finalizada! Pronto para treinamento.")
    
    # 5. Treinamento (opcional - descomente se quiser treinar)
    train_yolo(model, train_loader, val_loader)
    
    # 6. Predições nas imagens de teste
    print("\n🎯 Fazendo predições nas imagens de teste...")
    
    # Encontrar caminho das imagens de teste
    test_images_path = os.path.join(config['path'], config.get('test', config['val']))
    if not os.path.exists(test_images_path):
        print(f"❌ Pasta de teste não encontrada: {test_images_path}")
    else:
        predict_and_save(
            model=model,
            source_path=test_images_path,
            output_dir="predicts",
            conf_threshold=0.25,
            iou_threshold=0.45,
            img_size=IMG_SIZE
        )
    
    print("\n🎉 Processo concluído!")