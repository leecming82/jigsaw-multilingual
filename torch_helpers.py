import os
import torch
import re
from transformers import WEIGHTS_NAME, CONFIG_NAME

class EMA:
    """
    Tracks registered param data values with shadow variable that does
    exponential moving average of the value on update
    """

    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()


def save_model(output_dir, model, config, tokenizer):
    """ Save HuggingFace model to specified output dir """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model.state_dict(), output_model_file)
    config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_dir)


def layerwise_lr_decay(model, base_lr, decay_factor):
    # layerwise (at transformer block level) decay of LR
    decayed_lr_params = []
    max_num = None
    for name, param in reversed(list(model.named_parameters())):
        try:
            if 'base_model' not in name:  # classifier layers
                block_lr = base_lr
            else:
                if 'layer' in name:
                    block_num = int(re.findall(r'\d+', name)[0]) + 1
                    if max_num is None:
                        max_num = block_num
                else:  # typically embeddings
                    block_num = 0

                block_lr = base_lr * decay_factor ** (max_num - block_num)
        except:  # for base_model without number that aren't embeddings - pooler at the head
            block_lr = base_lr

        # print(name, block_lr)
        decayed_lr_params.append({'params': param,
                                  'lr': block_lr})
    return decayed_lr_params
