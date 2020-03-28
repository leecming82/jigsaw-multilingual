import os
import torch
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
    tokenizer.save_vocabulary(output_dir)