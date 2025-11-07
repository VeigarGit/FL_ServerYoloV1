import socket
import pickle
import struct
import torch
import threading
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
import time
import argparse
import sys
import traceback
import os
import h5py
from ultralytics import YOLO
from pathlib import Path

class FederatedLearningServer:
    def __init__(self, args):
        self.args = args
        if args.dataset == 'COCO128':
            self.global_model = YOLO("yolo11n.pt")
        self.rs_test_acc = []
        self.rs_test_loss = []
        self.global_state = self.global_model.state_dict()
        self.lock = threading.Lock()
        self.client_data = {}
        self.client_connections = []
        self.client_addresses = []
        self.client_idx = []
        self.clients_info = {}
        self.prune = args.prune
        self.device = torch.device(args.device)
        self.test_loader = None

    def aggregate_models(self, model_list):
        agg_state = {}
        for key in model_list[0].keys():
            agg_state[key] = sum([m[key] for m in model_list]) / len(model_list)
        return agg_state

    def send_data(self, conn, data):
        data_bytes = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(data_bytes)))
        conn.sendall(data_bytes)

    def recv_data(self, conn):
        raw_msglen = self.recvall(conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        data_bytes = self.recvall(conn, msglen)
        return pickle.loads(data_bytes)

    def recvall(self, conn, n):
        data = b'' 
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def handle_client(self, conn, client_updates, round_num, client_id):
        try:
            start_time = time.time()
            print(f"Round {round_num}: Handling client {client_id}")
            
            with self.lock:
                current_global_state = self.global_state.copy()
                size_before = sys.getsizeof(pickle.dumps(current_global_state)) / (1024 * 1024)  # MB
                keys = list(current_global_state.keys())
                #print("Keys before pruning:", keys)
                
                # Filtrar chaves que NÃO correspondem ao padrão model.model.23.*
                filtered_global_state = {k: v for k, v in current_global_state.items() 
                                    if not (k.startswith('model.model.23.') or k.startswith('model.model.10.'))}
                quantized_state_dict = {}
                for k, v in filtered_global_state.items():
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
                removed_keys = [k for k in keys if k.startswith('model.model.23.') or k.startswith('model.model.10.')]
                if removed_keys:
                    #print(f"Removendo camadas: {removed_keys}")
                    print(f"Total de camadas removidas: {len(removed_keys)}")
                size_after = sys.getsizeof(pickle.dumps(quantized_state_dict)) / (1024 * 1024)  # MB
    
                # Calcular economia
                size_saved = size_before - size_after
                print(f"Tamanho antes: {size_before:.2f} MB")
                print(f"Tamanho depois: {size_after:.2f} MB")
                print(f"Economia: {size_saved:.2f} MB")
                #print("Keys after pruning:", list(filtered_global_state.keys()))

            self.send_data(conn, quantized_state_dict)  # Envia o estado filtrado
            #self.send_data(conn, current_global_state)
            print(f"Round {round_num}: Sent global model to client {client_id}")
            
            updated_state = self.recv_data(conn)
            dequantized_state_dict = {}
            for k, v in updated_state.items():
                if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
                    # Recupera tensores quantizados
                    scale = v['scale']
                    dequantized_state_dict[k] = v['weights'].float() * scale
                else:
                    # Mantém tensores normais
                    dequantized_state_dict[k] = v
            self.client_data[client_id] = self.recv_data(conn)
            self.argalgo = self.recv_data(conn)
            end_time = time.time()
            
            if updated_state is not None:
                with self.lock:
                    client_updates.append(dequantized_state_dict)
                training_time = end_time - start_time
                self.clients_info[client_id]['training_time'] = training_time
                print(f"Round {round_num}: Client {client_id} training completed in {training_time:.2f} seconds")
            else:
                print(f"Round {round_num}: No update received from client {client_id}")
                
        except Exception as e:
            print(f"Round {round_num}: Error handling client {client_id}: {e}")
            traceback.print_exc()

    def load_test_data_yolo_ultralytics(self, client_id):
        dataset_path = Path("../dataset/COCO128/")
        client_yaml_path = dataset_path / f'{client_id}.yaml'
        
        if not client_yaml_path.exists():
            raise FileNotFoundError(f"YAML file for client {client_id} not found: {client_yaml_path}")
        
        return str(client_yaml_path)

    def evaluate_model_yolo_ultralytics(self, yaml_path):
        model = copy.deepcopy(self.global_model)
        results = model.val(data=yaml_path, imgsz=640, batch=16)
        map50 = results.box.map50
        map = results.box.map
        return map50, map

    def save_results(self):
        if self.args.prune == 0:
            a = "prune"
        else:
            a = "withou_Prune"
        b = self.argalgo
        if b == 0:
            b = "FedALA"
        else:
            b = "FedAVG"
        algo = self.args.dataset + "_" + a + "_" + b
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        file_path = result_path + "{}.h5".format(algo)
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_test_loss)

    def run_server(self):
        print("=== Federated Learning Server ===")
        print(f"Host: {self.args.host}:{self.args.port}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Clients per round: {self.args.clients_per_round}")
        print(f"Total rounds: {self.args.rounds}")
        print("=" * 40)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.args.host, self.args.port))
            s.listen(self.args.max_clients)
            print(f"Server listening on {self.args.host}:{self.args.port}")
            print(f"Waiting for {self.args.clients_per_round} clients to connect...")
            
            self.client_data = {index: None for index in range(1, self.args.clients_per_round+1)}
            while len(self.client_connections) < self.args.clients_per_round:
                conn, addr = s.accept()
                idx = self.recv_data(conn)
                print("client idx:", idx) 
                self.clients_info[idx+1] = {'training_time': None}
                print(f"Client {len(self.client_connections) + 1} connected: {addr}")
                self.client_idx.append(idx)
                self.client_connections.append(conn)
                self.client_addresses.append(addr)
            
            print(f"All {self.args.clients_per_round} clients connected. Starting training...")
            
            for round_num in range(self.args.rounds):
                print(f"\n--- Round {round_num + 1}/{self.args.rounds} ---")
                client_updates = []
                threads = []
                
                for i, conn in enumerate(self.client_connections):
                    t = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, client_updates, round_num + 1, i + 1)
                    )
                    t.start()
                    threads.append(t)
                
                for t in threads:
                    t.join()
                
                if client_updates:
                    print(f"Round {round_num + 1}: Aggregating {len(client_updates)} client updates")
                    aggregated_state = self.aggregate_models(client_updates)
                    
                    with self.lock:
                        self.global_state = aggregated_state
                        self.global_model.load_state_dict(self.global_state)
                    
                    if self.args.dataset == 'COCO128':
                        acc = []
                        for i in self.client_idx:
                            try:
                                yaml_path = self.load_test_data_yolo_ultralytics(i)
                                accuracy, avg_loss = self.evaluate_model_yolo_ultralytics(yaml_path)
                                acc.append(accuracy)
                            except Exception as e:
                                print(f"Error evaluating client {i}: {e}")
                                acc.append(0)
                        
                        if acc:
                            accuracy = sum(acc) / len(acc)
                            self.rs_test_acc.append(accuracy)
                            print(f"Round {round_num + 1}: Test mAP@0.5: {accuracy:.4f}")
                    
                    successful_notifications = 0
                    for conn in self.client_connections:
                        try:
                            conn.send('end'.encode('utf-8'))
                            successful_notifications += 1
                        except Exception as e:
                            print(f"Error notifying client: {e}")
                    
                    print(f"Round {round_num + 1}: Global model updated. Notified {successful_notifications} clients.")
                else:
                    print(f"Round {round_num + 1}: No client updates received this round.")
            
            print(f"\nTraining completed after {self.args.rounds} rounds!")
            
            for conn in self.client_connections:
                try:
                    conn.close()
                except:
                    pass
            print("All client connections closed.")
        self.save_results()

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--clients-per-round', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='COCO128', choices=['COCO128'])
    parser.add_argument('--test-client-idx', type=int, default=0)
    parser.add_argument('--max-clients', type=int, default=10)
    parser.add_argument('--prune', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', '--device_id', type=str, default="0")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda is not available.")
        args.device = "cpu"
    
    server = FederatedLearningServer(args)
    server.run_server()

if __name__ == '__main__':
    main()