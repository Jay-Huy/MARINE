from transformers import (LogitsProcessor)
import torch
import torch.nn.functional as F

class GuidanceLogits(LogitsProcessor):

    def __init__(self, guidance_strength, guidance, images, attention_mask, model, tokenizer=None, attention_catcher=None):
        """
        Args:
            guidance_strength (float): The guidance strength for the logits.
            guidance (torch.Tensor): The guidance input tensor.
            images (torch.Tensor): The images tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            model (torch.nn.Module): The model to use.
            attention_catcher (AttentionCatcher): Hook để lấy Uncondition Attention.
        """
        self.guidance_strength = guidance_strength
        self.guidance = guidance.cuda()
        self.images = images
        self.attention_mask = attention_mask
        self.model = model
        self.out = None
        self.tokenizer = tokenizer
        self.attention_catcher = attention_catcher
        self.sam_mask = kwargs.get('sam_mask', None)
        self.is_uncond_pass = False
        self.tau = kwargs.get('tau', 2.8)
        self.beta = kwargs.get('beta', 3.0)
        self.gamma_scale = 1.0
        
        # Register pre_hooks on all self_attn layers to apply Soft Spatial Penalty
        self.pre_hook_handles = []
        for layer in self.model.language_model.model.layers:
            handle = layer.self_attn.register_forward_pre_hook(self.pre_hook_fn, with_kwargs=True)
            self.pre_hook_handles.append(handle)
        
    def pre_hook_fn(self, module, args, kwargs):
        if not self.is_uncond_pass or self.sam_mask is None:
            return args, kwargs
            
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            kv_len = attention_mask.shape[-1]
            image_idx = (self.guidance[0] == -200).nonzero(as_tuple=True)[0].item()
            
            if kv_len >= image_idx + 576:
                # self.sam_mask chứa giá trị 1.0 (foreground) và alpha (background).
                # Tính penalty = log(sam_mask). 
                # Với foreground: log(1.0) = 0 (giữ nguyên).
                # Với background: log(alpha) = số âm (hạ attention_score).
                penalty_1d = torch.log(self.sam_mask + 1e-9) # (batch, 576)
                
                penalty = torch.zeros((self.sam_mask.shape[0], 1, 1, kv_len), device=attention_mask.device, dtype=attention_mask.dtype)
                penalty[:, 0, 0, image_idx : image_idx + 576] = penalty_1d.to(penalty.dtype)
                
                kwargs['attention_mask'] = attention_mask + penalty
                
        return args, kwargs

    def clean_up(self):
        """Xóa toàn bộ hooks để tránh memory leak sau mỗi batch."""
        for handle in self.pre_hook_handles:
            handle.remove()
        self.pre_hook_handles = []

    def get_dynamic_gamma(self):
        """
        Tính Entropy H từ Uncondition Attention và suy ra dynamic gamma bằng Sigmoid.
        """
        if self.attention_catcher is None or self.attention_catcher.attention is None:
            return self.guidance_strength # Fallback về Static Gamma nếu không bắt được attention

        uncond_attn = self.attention_catcher.attention
        # uncond_attn shape: (batch_size, num_heads, q_len, kv_len)
        # Tại mỗi bước sinh token, q_len = 1 (chỉ query cho token hiện tại)
        # Ta lấy attention của batch 0, tính trung bình qua các heads
        attention_weights = uncond_attn[0, :, -1, :] # (num_heads, kv_len)
        mean_attention = attention_weights.mean(dim=0)
        
        # Tính Entropy
        H = -torch.sum(mean_attention * torch.log(mean_attention + 1e-9)).item()
        
        # Áp dụng Sigmoid
        gamma = self.gamma_scale * torch.sigmoid(torch.tensor(self.beta * (H - self.tau)))
        return gamma.item()

    def __call__(self, input_ids, logits):
        logits = F.log_softmax(logits, dim=-1)
        self.is_uncond_pass = True # Bật cờ để kích hoạt Soft Spatial Penalty
        
        if self.out is None:
            self.out = self.model(input_ids=self.guidance, 
                                  pixel_values=self.images, 
                                  attention_mask=self.attention_mask,
                                  use_cache=True,
                                  output_attentions=True)
        else:
            self.out = self.model(input_ids[:, -1:],
                                  use_cache=True,
                                  attention_mask=self.attention_mask, # Giữ nguyên mask cho các step sau
                                  past_key_values=self.out.past_key_values,
                                  output_attentions=True)
                                  
        self.is_uncond_pass = False # Tắt cờ

        if len(self.out.logits) == 1:
            guidance_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        else:
            guidance_logits = F.log_softmax(self.out.logits[:,-1:], dim=-1).to(logits.device)
            guidance_logits = guidance_logits.squeeze(1)

        # Tính Dynamic Gamma dựa trên Uncondition Attention thay vì dùng biến tĩnh
        dynamic_gamma = self.get_dynamic_gamma()

        out = dynamic_gamma * (guidance_logits - logits) + logits
        out = F.log_softmax(out, dim=-1)

        return out