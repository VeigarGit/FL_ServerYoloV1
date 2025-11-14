import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import math

def prune_and_restructure(model, pruning_rate=0.5, n_in=3, size_fc=25):
    model_copy = copy.deepcopy(model)
    layers = []
    indices_not_remove_weight = None
    first_conv = True
    first_linear = True
    index_last_layer = 0
    masks = []

    # Encontrar índice da última camada linear
    for n, module in enumerate(model_copy.modules()):
        if isinstance(module, nn.Linear):
            index_last_layer = n

    for n, module in enumerate(model_copy.modules()):
        if isinstance(module, nn.Conv2d):
            print(f"Processando Conv2d: {module.in_channels}->{module.out_channels}, groups={module.groups}")
            
            # Realiza o pruning e substitui os pesos por 0
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
            masks.append(next(model_copy.buffers()))
            prune.remove(module, 'weight')

            # Filtra onde não é 0 (neurônios de saída)
            indices_not_remove_perceptron = module.weight.abs().sum(dim=(1, 2, 3)) != 0

            # Aplica a filtragem dos valores 0
            weight_prunned = module.weight[indices_not_remove_perceptron]

            # Filtra os pesos da camada anterior (canais de entrada)
            if not first_conv and indices_not_remove_weight is not None:
                if weight_prunned.size(1) == len(indices_not_remove_weight):
                    weight_prunned = weight_prunned[:, indices_not_remove_weight, :, :]
                
            if module.bias is not None:
                bias_prunned = module.bias[indices_not_remove_perceptron]
            else:
                bias_prunned = None

            # Determina tamanho da saída
            n_out = sum(indices_not_remove_perceptron).item()
            
            # CORREÇÃO CRÍTICA: Ajustar grupos para compatibilidade com in_channels E out_channels
            if module.groups > 1:
                if module.groups == module.in_channels and module.groups == module.out_channels:
                    # Depthwise convolution - grupos deve ser igual a ambos
                    new_groups = min(n_in, n_out)
                else:
                    # Encontrar divisor comum entre n_in e n_out
                    gcd_value = math.gcd(n_in, n_out)
                    if gcd_value > 1:
                        new_groups = gcd_value
                    else:
                        new_groups = 1  # Fallback para convolution normal
                
                # Garantir que ambos são divisíveis pelos grupos
                if n_in % new_groups != 0:
                    new_groups = math.gcd(n_in, new_groups)
                if n_out % new_groups != 0:
                    new_groups = math.gcd(n_out, new_groups)
                
                # Último fallback
                if new_groups == 0 or n_in % new_groups != 0 or n_out % new_groups != 0:
                    new_groups = 1
                    
                print(f"  Ajustando grupos: {module.groups} -> {new_groups}")
            else:
                new_groups = 1

            print(f"  Nova Conv2d: in={n_in}, out={n_out}, groups={new_groups}")

            # Cria nova camada com os parâmetros filtrados
            new_layer = nn.Conv2d(
                in_channels=n_in,
                out_channels=n_out,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=new_groups,
                bias=(module.bias is not None)
            )
            
            # Atribui os pesos pruned
            with torch.no_grad():
                new_layer.weight.data = weight_prunned
                if bias_prunned is not None:
                    new_layer.bias.data = bias_prunned

            layers.append(new_layer)
            indices_not_remove_weight = indices_not_remove_perceptron
            n_in = n_out
            first_conv = False

        elif isinstance(module, nn.Linear):            
            print(f"Processando Linear: {module.in_features}->{module.out_features}")
            
            # Realiza o pruning e substitui os pesos por 0
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
            masks.append(next(model_copy.buffers()))
            prune.remove(module, 'weight')

            # Filtra onde não é 0 (neurônios de saída)
            indices_not_remove_perceptron = module.weight.abs().sum(dim=1) != 0

            # Primeira camada linear - precisa conectar com a última conv
            if first_linear:
                layers.append(nn.Flatten())
                first_linear = False
                
                if indices_not_remove_weight is not None:
                    n_in = sum(indices_not_remove_weight).item()
                else:
                    n_in = module.in_features
            
            # Remove os pesos que não serão utilizados (conexões de entrada)
            if indices_not_remove_weight is not None:
                if module.weight.size(1) == len(indices_not_remove_weight):
                    weight_prunned = module.weight[:, indices_not_remove_weight]
                else:
                    weight_prunned = module.weight
            else:
                weight_prunned = module.weight

            # Não reestrutura a última camada
            if n == index_last_layer:
                indices_not_remove_perceptron = torch.ones(module.weight.size(0), dtype=torch.bool)
                weight_prunned = weight_prunned
            
            weight_prunned = weight_prunned[indices_not_remove_perceptron]
                
            if module.bias is not None:
                bias_prunned = module.bias[indices_not_remove_perceptron]
            else:
                bias_prunned = None
                
            n_out = sum(indices_not_remove_perceptron).item()

            new_layer = nn.Linear(n_in, n_out)
            
            with torch.no_grad():
                new_layer.weight.data = weight_prunned
                if bias_prunned is not None:
                    new_layer.bias.data = bias_prunned
            
            layers.append(new_layer)
            indices_not_remove_weight = indices_not_remove_perceptron
            n_in = n_out

        elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.Dropout, nn.SiLU, nn.Hardswish)):
            # Adiciona várias funções de ativação comuns no YOLO
            layers.append(module)

    new_model = nn.Sequential(*layers)
    print("Pruning e reestruturação concluídos!")
    return new_model, masks