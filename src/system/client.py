import socket
import pickle
import struct
import torch
import argparse
import os
from ultralytics import YOLO
from pathlib import Path
import copy
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
    dataset_path = Path("../dataset/COCO128/")
    client_yaml_path = dataset_path / f'{client_id}.yaml'
    
    if not client_yaml_path.exists():
        raise FileNotFoundError(f"YAML file for client {client_id} not found: {client_yaml_path}")
    
    return str(client_yaml_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='COCO128', choices=['COCO128'])
    parser.add_argument('--client-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', '--device_id', type=str, default="0")
    
    return parser.parse_args()

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
    
    model = YOLO("yolo11n.pt")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((args.host, args.port))
            send_data(s, args.client_idx)
            print(f"Connected to server {args.host}:{args.port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            return
        
        for round_num in range(args.rounds):
            print(f"\n--- Round {round_num + 1}/{args.rounds} ---")
            
            global_state = recv_data(s)
            if global_state is None:
                print("Failed to receive global model. Connection may be closed.")
                break
            
            print("Received global model.")
            
            state = type(model)()
            state.load_state_dict(copy.deepcopy(model.state_dict()))
            state.load_state_dict(global_state)
            set_parameters(model, state)
            
            try:
                yaml_path = load_train_data_yolo_ultralytics(args.client_idx)
                model.train(data=yaml_path, epochs=1, batch=args.batch_size, imgsz=640, device=device)
                print("Local training completed.")
            except Exception as e:
                print(f"Training failed: {e}")
                break
            
            updated_state = model.state_dict()
            
            send_data(s, updated_state)
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

if __name__ == '__main__':
    main()