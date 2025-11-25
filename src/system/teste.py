from ultralytics import YOLO
from pathlib import Path
import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from ultralytics import YOLO
from prunningyolo import prune_and_restructure
from size_mode import get_model_size
import tempfile

def set_parameters(model, state_new):
        for new_param, old_param in zip(state_new.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
# Load a COCO-pretrained YOLO11n model
model = YOLO("myyolo2.yaml")  # build a new model from YAML
print(get_model_size(model))
selected_class_ids = [0, 1, 2, 3, 5, 7]  # Exemplo de IDs de classes selecionadas
yaml_path = Path("../dataset/P-All/")
client_yaml_path = yaml_path / f'P-All.yaml'
client_yaml_path = str(client_yaml_path)
#model.export(format='engine', half=True, int8=True, data=client_yaml_path,batch=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train(data=client_yaml_path, epochs=60, batch=8, imgsz=640, device=device, patience=64, save_period=5,classes=selected_class_ids,val=True,plots=True, verbose=False,save_json=True)
model.val(data=client_yaml_path, imgsz=640, batch=8, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)
model.export(format='engine', half=True, int8=True, data=client_yaml_path,batch=8)
model.val(data=client_yaml_path, imgsz=640, batch=8, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)



'''
pruned_model, _ = prune_and_restructure(model, pruning_rate=0.5)

set_parameters(model, pruned_model)
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
    temp_path = tmp_file.name
    # Salvar o modelo local no arquivo temporário
    model.save(temp_path)
    # Carregar o modelo a partir do arquivo temporámodel.train(data=client_yaml_path, epochs=40, batch=8, imgsz=640, device=device, patience=100, save_period=5,classes=selected_class_ids,val=True,plots=True, verbose=False,save_json=True)
model.val(data=client_yaml_path, imgsz=640, batch=8, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)
model.export(format='engine', half=True, int8=True, data=client_yaml_path,batch=8)
model.val(data=client_yaml_path, imgsz=640, batch=8, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)rio

mod = YOLO(temp_path)
print(get_model_size(mod))
#mod.save("hentai.pt")model.train(data=client_yaml_path, epochs=40, batch=8, imgsz=640, device=device, patience=100, save_period=5,classes=selected_class_ids,val=True,plots=True, verbose=False,save_json=True)
model.val(data=client_yaml_path, imgsz=640, batch=8, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)
model.export(format='engine', half=True, int8=True, data=client_yaml_path,batch=8)
model.val(data=client_yaml_path, imgsz=640, batch=8, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)
selected_class_ids = [0, 1, 2, 3, 5, 7]  # Exemplo de IDs de classes selecionadas
yaml_path = Path("../dataset/tcl/")
client_yaml_path = yaml_path / f'1.yaml'
client_yaml_path = str(client_yaml_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mod.train(data=client_yaml_path, epochs=40, batch=4, imgsz=640, device=device, patience=100, save_period=5,classes=selected_class_ids,val=True,plots=True, verbose=False,save_json=True)
client_yaml_path = yaml_path / f'0.yaml'
client_yaml_path = str(client_yaml_path)
results = mod.val(data=client_yaml_path, imgsz=640, batch=2, device=device, classes=selected_class_ids, plots=True, save_json=True, verbose=False)
print(results.box.map50)
print(results.box.map)
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
    temp_path = tmp_file.name
    # Salvar o modelo local no arquivo temporário
    mod.save(temp_path)
mid = YOLO(temp_path)
mid.save('modelo_prunned.pt')
torch.save(mid.model.state_dict(), 'pesos_prunned.pth')
#model.train(data=client_yaml_path, epochs=10, batch=2, imgsz=640, device=device, patience=100, save_period=5,classes=selected_class_ids,val=True,plots=True, verbose=False,save_json=True)
'''