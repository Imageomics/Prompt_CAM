import math
import torch
import torch.nn as nn


def init_weight(down, up, option):
    with torch.no_grad():
        if option == 'lora_kaiming':
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            nn.init.zeros_(up.weight)
            nn.init.zeros_(down.bias)
            nn.init.zeros_(up.bias)
        elif option == 'lora_xavier':
            nn.init.xavier_uniform_(down.weight)
            nn.init.zeros_(up.weight)
            nn.init.zeros_(down.bias)
            nn.init.zeros_(up.bias)
        elif option == 'xavier':
            nn.init.xavier_uniform_(down.weight)
            nn.init.xavier_uniform_(up.weight)
            nn.init.normal_(down.bias, std=1e-6)
            nn.init.normal_(up.bias, std=1e-6)
        elif option == 'zero':
            nn.init.zeros_(down.weight)
            nn.init.zeros_(up.bias)
            nn.init.zeros_(down.weight)
            nn.init.zeros_(up.bias)
        else:
            raise NotImplementedError