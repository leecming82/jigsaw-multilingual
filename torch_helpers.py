import os
import torch
import re
from transformers import WEIGHTS_NAME, CONFIG_NAME


def mask_tokens(inputs, tokenizer, mlm_prob=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    pad_tokens_mask = [[1 if x == tokenizer.pad_token_id else 0 for x in val] for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    probability_matrix.masked_fill_(torch.tensor(pad_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).cuda()
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

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
