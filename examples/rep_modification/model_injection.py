# wrapping classes
import torch
import numpy as np

class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.controller = None
        self.token_pos = None
        self.mod_method = 'add'

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if self.controller is not None:
            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            assert len(self.controller.shape) == len(output.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."
            self.controller = self.controller.to(modified.device)
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = modified[:, self.token_pos] + self.controller * mask
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                modified[:, self.token_pos] = modified[:, self.token_pos] + self.controller * mask
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    len_token = self.controller.shape[1]
                    modified[:, -len_token:] = modified[:, -len_token:] + self.controller * mask
                elif self.token_pos == "start":
                    len_token = self.controller.shape[1]
                    modified[:, :len_token] = modified[:, :len_token] + self.controller * mask
                else:
                    assert False, f"Unknown token position {self.token_pos}."
            else:
                modified = modified + self.controller

        return output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False):
        self.normalize = normalize
        self.controller = activations.squeeze()
        self.mask = masks
        self.token_pos = token_pos
        
    def reset(self):
        self.output = None
        self.controller = None
        self.mask = None

    def set_masks(self, masks):
        self.mask = masks

    
class WrappedReadingVecModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, prompt, max_new_tokens=100, random_seed=0, use_cache=True):
        with torch.no_grad():
            torch.random.manual_seed(random_seed)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            attention_mask = inputs.attention_mask.to(self.model.device)
            generate_ids = self.model.generate(inputs.input_ids.to(self.model.device), attention_mask=attention_mask, max_new_tokens=max_new_tokens, use_cache=use_cache)
            return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits
        
    def run_prompt(self, prompt, **kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output
        
    def wrap_decoder_block(self, layer_id):
        block = self.model.model.layers[layer_id]
        if not self.is_wrapped(block):
            self.model.model.layers[layer_id] = WrappedBlock(block)
    
    def wrap_all(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            self.wrap_decoder_block(layer_id)
            
    def wrap_block(self, layer_ids, block_name):
        def _wrap_block(layer_id, block_name):
            self.wrap_decoder_block(layer_id)
            
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)

            
    def get_activations(self, layer_ids, block_name='decoder_block'):

        def _get_activations(layer_id, block_name):
            current_layer = self.model.model.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                return current_layer.output
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            activations = {}
            for layer_id in layer_ids:
                activations[layer_id] = _get_activations(layer_id, block_name)
            return activations
        else:
            return _get_activations(layer_ids, block_name)

    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None, normalize=False):

        def _set_controller(layer_id, activations, block_name, masks, normalize):
            current_layer = self.model.model.layers[layer_id]
            current_layer.set_controller(activations, token_pos, masks, normalize)
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            assert isinstance(activations, dict), "activations should be a dictionary"
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[layer_id], block_name, masks, normalize)
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize)
      
    def reset(self):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.reset()

    def set_masks(self, masks):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.set_masks(masks)

    def is_wrapped(self, block):
        if hasattr(block, 'block'):
            return True
        return False
    
    def unwrap(self):
        for l, layer in enumerate(self.model.model.layers):
            if self.is_wrapped(layer):
                self.model.model.layers[l] = layer.block
