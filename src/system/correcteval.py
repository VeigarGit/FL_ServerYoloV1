from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
from ultralytics.utils.metrics import ConfusionMatrix

model_file = ".pt"
yaml_file = ".yaml"

model = YOLO(model_file)

selected_class_ids=[0,1,2,3,5,7]
selected_class_names = [model.names[idx] for idx in selected_class_ids]
names_dict = {i: name for i, name in enumerate(selected_class_names)}

results = model.val(
    data=yaml_file,
    split='test',
    save_json=True,
    device=0,
    plots=True,
    classes=selected_class_ids
)

original_matrix = results.confusion_matrix.matrix
background_idx = original_matrix.shape[0] - 1
selected_indices = list(selected_class_ids) + [background_idx]
# Filter the matrix including background
filtered_matrix = original_matrix[selected_indices][:, selected_indices]
# Update the names to include background
filtered_conf_matrix = ConfusionMatrix(names=names_dict)
filtered_conf_matrix.matrix = filtered_matrix
filtered_conf_matrix.plot(save_dir="normalized", normalize=True)
filtered_conf_matrix.plot(save_dir="raw", normalize=False)