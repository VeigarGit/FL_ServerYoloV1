import socket
import pickle
import struct
import torch
import argparse
import os
from ultralytics import YOLO
from pathlib import Path
import copy
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
from torch.utils.data import Subset
from PIL import Image
from torchvision import transforms
import sys
import h5py
import tempfile
from size_mode import get_model_size
def send_data(conn, data):
    data_bytes = pickle.dumps(data)
    conn.sendall(struct.pack('!I', len(data_bytes)))
    conn.sendall(data_bytes)

def recv_data(conn):
    raw_msglen = recvall(conn, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!I', raw_msglen)[0]
    data_bytes = recvall(conn, msglen)
    return pickle.loads(data_bytes)

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def set_parameters(model, state_new):
    for new_param, old_param in zip(state_new.parameters(), model.parameters()):
        old_param.data = new_param.data.clone()

def load_train_data_yolo_ultralytics(client_id):
    dataset_path = Path("../dataset/tcl/")
    client_yaml_path = dataset_path / f'{client_id}.yaml'
    
    if not client_yaml_path.exists():
        raise FileNotFoundError(f"YAML file for client {client_id} not found: {client_yaml_path}")
    
    return str(client_yaml_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='COCO128', choices=['COCO128'])
    parser.add_argument('--client-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', '--device_id', type=str, default="0")
    
    return parser.parse_args()
class CustomImageDataset():
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        
        # DEFINIR TRANSFORMAÇÃO OBRIGATÓRIA que inclui ToTensor()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((416, 416)),
                transforms.ToTensor(),  # ESSENCIAL: converte PIL para Tensor
            ])
        else:
            self.transform = transform
            
        # Listar apenas arquivos de imagem
        self.images = [item for item in os.listdir(img_dir) 
                      if os.path.isfile(os.path.join(img_dir, item)) and
                      item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Encontradas {len(self.images)} imagens em {img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        
        try:
            # Carregar imagem
            image = Image.open(img_path).convert('RGB')
            
            # APLICAR A TRANSFORMAÇÃO (inclui ToTensor())
            if self.transform:
                image = self.transform(image)
            else:
                # Fallback: converter para tensor manualmente se não htransform
                image = transforms.ToTensor()(image)
            
            # Verificar se a imagem é um tensor
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Imagem não é um tensor: {type(image)}")
            
            # Placeholder para targets (ajuste conforme seu dataset)
            # Para COCO, você precisaria carregar os labels reais
            target = torch.zeros(1, 5)  # [class_id, x_center, y_center, width, height]
            
            return image, target
            
        except Exception as e:
            print(f"Erro ao carregar {img_path}: {e}")
            # Retornar tensores dummy em caso de erro
            dummy_image = torch.zeros(3, 416, 416)
            dummy_target = torch.zeros(1, 5)
            return dummy_image, dummy_target

def adaptive_local_aggregation(client_id, device,
                            global_model: nn.Module,
                            local_model: nn.Module,
                            mask=None) -> None:
    """
    Versão corrigida do ALA usando state_dict() em vez de deepcopy
    """
    
    print(f"Iniciando ALA para cliente {client_id}")
    
    # Mover modelos para o device
    global_model = global_model.to(device)
    local_model = local_model.to(device)
    
    # Obter listas de parâmetros
    params_g = list(global_model.parameters())
    params = list(local_model.parameters())
    
    if len(params_g) == 0 or len(params) == 0:
        print("Erro: Modelos não têm parâmetros")
        return
    
    # Verificar se é primeira iteração
    if torch.sum(params_g[0] - params[0]) == 0:
        print("Primeira iteração - ALA desativado")
        return
    
    # 1. Preservar updates nas camadas inferiores
    print("Preservando camadas inferiores...")
    for param, param_g in zip(params[:-2], params_g[:-2]):
        param.data.copy_(param_g.data)
    
    # 2. Criar modelo temporário usando state_dict (evita deepcopy)
    #model_t = type(local_model)()  # Criar nova instância do mesmo tipo
    #model_t.load_state_dict(local_model.state_dict())
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        temp_path = tmp_file.name
    # Salvar o modelo local no arquivo temporário
    local_model.save(temp_path)
    # Carregar o modelo a partir do arquivo temporário
    model_t = YOLO(temp_path)
    model_t = model_t.to(device)
    params_t = list(model_t.parameters())
    
    # 3. Focar apenas nas camadas superiores
    if len(params) < 2:
        print("Modelo tem muito poucas camadas para ALA")
        return
        
    params_p = params[-2:]  # Parâmetros locais (últimas 2 camadas)
    params_gp = params_g[-2:]  # Parâmetros globais (últimas 2 camadas)  
    params_tp = params_t[-2:]  # Parâmetros temporários (últimas 2 camadas)
    
    # 4. Congelar camadas inferiores no modelo temporário
    for param in params_t[:-2]:
        param.requires_grad = False
    
    # 5. Inicializar pesos
    weights = [torch.ones_like(param.data).to(device) for param in params_p]
    
    # 6. Inicializar camadas superiores no modelo temporário
    for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, weights):
        param_t.data.copy_(param + (param_g - param) * weight)
    
    # 7. SIMULAÇÃO DE TREINO - Criar gradientes artificiais
    print("Simulando treino para obter gradientes...")
    
    # Para cada parâmetro nas camadas superiores, criar gradientes simulados
    for i, (param_t, param, param_g, weight) in enumerate(zip(params_tp, params_p, params_gp, weights)):
        # Garantir que param_t requer gradientes
        param_t.requires_grad = True
        
        # Simular forward pass
        x = param_t.data.clone().requires_grad_(True)
        target = torch.randn_like(x)
        
        # Calcular loss simulada
        loss = torch.nn.functional.mse_loss(x, target)
        
        # Backward para gerar gradientes
        loss.backward()
        
        # Usar o gradiente simulado para atualizar os pesos
        if x.grad is not None:
            with torch.no_grad():
                # Atualizar peso
                weight_update = x.grad * (param_g - param)
                weight.data.copy_(torch.clamp(weight - 0.1 * weight_update, 0, 1))
                
                # Atualizar parâmetro temporário
                param_t.data.copy_(param + (param_g - param) * weight)
        
        print(f"Camada {i}: peso médio = {weight.mean().item():.4f}")
    
    # 8. Aplicar mudanças ao modelo local
    for param, param_t in zip(params_p, params_tp):
        param.data.copy_(param_t.data)
    
    print(f"ALA concluído para cliente {client_id}")

    # Limpar memória
    del model_t, params_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def evaluate_model_yolo_ultralytics(yaml_path, model_local, device):
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        temp_path = tmp_file.name
    # Salvar o modelo local no arquivo temporário
    model_local.save(temp_path)
    # Carregar o modelo a partir do arquivo temporário
    mod = YOLO(temp_path)
    selected_class_ids = [0, 1, 2, 3, 5, 7]
    results = mod.val(data=yaml_path, imgsz=640, batch=2, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)
    map50 = results.box.map50
    map = results.box.map
    return map50, map
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda is not available.")
        args.device = "cpu"
    
    device = torch.device(args.device)
    
    print("=== Federated Learning Client ===")
    print(f"Host: {args.host}:{args.port}")
    print(f"Dataset: {args.dataset}")
    print(f"Client: {args.client_idx}")
    print(f"Rounds: {args.rounds}")
    print(f"Device: {args.device}")
    print("=" * 40)
    rs_test_acc=[0]
    rs_test_loss=[0]
    
    model = YOLO("yolo11n.pt")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((args.host, args.port))
            send_data(s, args.client_idx)
            print(f"Connected to server {args.host}:{args.port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            return
        selected_class_ids = [0, 1, 2, 3, 5, 7]  # Exemplo de IDs de classes selecionadas
        yaml_path = load_train_data_yolo_ultralytics(args.client_idx)
        #model.train(data=yaml_path, epochs=1, batch=args.batch_size, imgsz=640, device=device, patience=100, save_period=5,classes=selected_class_ids,val=True,plots=True, verbose=False,save_json=True)
        local_state = model.state_dict()
        for round_num in range(args.rounds):
            print(f"\n--- Round {round_num + 1}/{args.rounds} ---")
            
            global_state = recv_data(s)
            if global_state is None:
                print("Failed to receive global model. Connection may be closed.")
                break

            print("Received global model.")
            dequantized_state_dict = {}
            for k, v in global_state.items():
                if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
                    # Recupera tensores quantizados
                    scale = v['scale']
                    dequantized_state_dict[k] = v['weights'].float() * scale
                else:
                    # Mantém tensores normais
                    dequantized_state_dict[k] = v

            #state = type(model)().to(device)
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                temp_path = tmp_file.name
            # Salvar o modelo local no arquivo temporário
            model.save(temp_path)
            # Carregar o modelo a partir do arquivo temporário
            state = YOLO(temp_path)
            #state.load_state_dict(local_state)
            missing_keys, unexpected_keys = state.load_state_dict(dequantized_state_dict, strict=False)
            get = get_model_size(state)
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                temp_path = tmp_file.name
            # Salvar o modelo local no arquivo temporário
            model.save(temp_path)
            # Carregar o modelo a partir do arquivo temporário
            model = YOLO(temp_path)
            
            #adaptive_local_aggregation(args.client_idx, device,state,model)
            #set_parameters(model, state)
            acc=[]
            loss=[]
            try:
                yaml_path = load_train_data_yolo_ultralytics(args.client_idx)
                accuracy, avg_loss = evaluate_model_yolo_ultralytics(yaml_path, model, device)
                acc.append(accuracy)
                loss.append(avg_loss)
            except Exception as e:
                print(f"Error evaluating client {args.client_idx}: {e}")
                acc.append(0)
                loss.append(0)

            if acc:
                accuracy = sum(acc) / len(acc)
                rs_test_acc.append(accuracy)
                print(f"Round {round_num + 1}: Test mAP@0.5: {accuracy:.4f}")
            if loss:
                losses = sum(loss) / len(loss)
                rs_test_loss.append(losses)
                print(f"Round {round_num + 1}: Test mAP@mAP50-95: {losses:.4f}")
            save_results(rs_test_acc,rs_test_loss, args.dataset, args.client_idx)      
            try:
                selected_class_ids = [0, 1, 2, 3, 5, 7]  # Exemplo de IDs de classes selecionadas
                yaml_path = load_train_data_yolo_ultralytics(args.client_idx)
                #if round_num > 0:
                model.train(data=yaml_path, epochs=1, batch=args.batch_size, imgsz=640, device=device, patience=100, save_period=5,classes=selected_class_ids,val=True,plots=True, verbose=False,save_json=True)
                print("Local training completed.")
            except Exception as e:
                print(f"Training failed: {e}")
                break
            
            updated_state = model.state_dict()
            keys = list(updated_state.keys())
            size_before = sys.getsizeof(pickle.dumps(updated_state)) / (1024 * 1024)  # MB
            filtered_updated_state = {k: v for k, v in updated_state.items() 
                                    if not (k.startswith('model.model.23'))}# or k.startswith('model.model.10.'))}
            quantized_state_dict = {}
            for k, v in updated_state.items():
                if v.dtype == torch.float32:
                    # Quantização para int8 (preservando escala e zero_point)
                    v_f32 = v.clone().float()
                    scale = v_f32.abs().max() / 127.0
                    quantized_tensor = torch.round(v_f32 / scale).clamp(-128, 127).to(torch.int8)
                    
                    # Armazena tensor + metadados de quantização
                    quantized_state_dict[k] = {
                        'weights': quantized_tensor,
                        'scale': scale,
                        'dtype': 'quantized_int8'
                    }
                else:
                    # Mantém tensores não-float32 originais
                    quantized_state_dict[k] = v
            # Opcional: verificar quais chaves foram removidas
            removed_keys = [k for k in keys if k.startswith('model.model.23') ]#or k.startswith('model.model.10.')]
            if removed_keys:
                #print(f"Removendo camadas: {removed_keys}")
                print(f"Total de camadas removidas: {len(removed_keys)}")
            size_after = sys.getsizeof(pickle.dumps(quantized_state_dict)) / (1024 * 1024)  # MB

            # Calcular economia
            size_saved = size_before - size_after
            print(f"Tamanho antes: {size_before:.2f} MB")
            print(f"Tamanho depois: {size_after:.2f} MB")
            print(f"Economia: {size_saved:.2f} MB")
            send_data(s, quantized_state_dict)
            send_data(s, 6)
            send_data(s, 1)
            print("Client update sent.")
            
            try:
                s.recv(3)
                print("Ready for next round...")
            except Exception as e:
                print(f"Error waiting for server: {e}")
                break
         
    print("\nTraining completed!")
def save_results(rs_test_acc,rs_test_loss, dataset, client_idx):
        dataset = str(dataset)
        a = str(client_idx)
        algo = dataset + "_client_" + a
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        file_path = result_path + "{}.h5".format(algo)
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('mAP50', data=rs_test_acc)
            hf.create_dataset('mAP50-95', data=rs_test_loss)
if __name__ == '__main__':
    main()
