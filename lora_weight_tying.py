import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import GPTNeoXForCausalLM


class LoraWeightTyingConfig(LoraConfig):
    """
    Lora config for weight tying (i.e., tying the lora weights of all 
    the layers of the model).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_weight_tying = True


def enable_weight_tying(model):
    """
    Enable weight tying for the specified model.
    """
    
    # Get the active adapters for every target module in the model.

    # Get the first layer lora modules
    lora_A = None
    lora_B = None


    # Check every parameter in the model.
    for name, param in model.named_parameters():
        
        # Check if the parameter is a lora weight
        if name.endswith("lora_A") or name.endswith("lora_B"):
            
            parts = name.split('.')
            parent_module = model
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)

            if lora_A is None:
                lora_A = parent_module.getattr(parts[-1])
                lora_B = parent_module.getattr(parts[-1])
            
            else:
                parent_module.setattr(parts[-1], lora_A)
                parent_module.setattr(parts[-1], lora_B)

    return model


def get_peft_model_weight_tying(base_model, config):
    """
    Get the PEFT model with the specified base model and config.
    """
    model = get_peft_model(base_model, config)

    if config.enable_weight_tying:
        model = enable_weight_tying(model)

    return model


if __name__ == '__main__':

    # Create a base model from Pythia
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m")

    # Create a config for weight tying
    config = LoraWeightTyingConfig(
        r=8, 
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_key_value"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get the PEFT model with weight tying
    model = get_peft_model_weight_tying(model, config)

    # Validate that weights are tied by running one forward and backward pass
    input_ids = torch.randint(0, 50256, (1, 512))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    print(f"Loss: {loss.item()}")

    loss.backward()

    # Check if the weights are tied
    first_layer_lora_A = model.model.gpt_neox.layers[0].attention.query_key_value.lora_A.default.weight
    first_layer_lora_B = model.model.gpt_neox.layers[0].attention.query_key_value.lora_B.default.weight

    for name, param in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            if not torch.equal(param, first_layer_lora_A) and not torch.equal(param, first_layer_lora_B):
                print(f"Weights are not tied for {name}!")
                break

    print("Weights are tied!")