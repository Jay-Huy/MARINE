from transformers import (LogitsProcessor)
import torch
import torch.nn.functional as F

class GuidanceLogits(LogitsProcessor):

    def __init__(self, guidance, images, attention_mask, model, tokenizer=None, attention_catcher=None, tau=2.8, beta=3.0, sam_mask=None):
        """
        Args:
            guidance (torch.Tensor): The guidance input tensor.
            images (torch.Tensor): The images tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            model (torch.nn.Module): The model to use.
            attention_catcher (AttentionCatcher): Hook để lấy Uncondition Attention.
            tau (float): Tau parameter for entropy mapping
            beta (float): Beta parameter for entropy mapping
            sam_mask (torch.Tensor): Mask tensor containing 1.0 and alpha
        """
        self.guidance = guidance.cuda()
        self.images = images
        self.attention_mask = attention_mask
        self.model = model
        self.out = None
        self.tokenizer = tokenizer
        self.attention_catcher = attention_catcher
        self.sam_mask = sam_mask
        self.is_uncond_pass = False
        self.tau = tau
        self.beta = beta
        self.gamma_scale = 1.0
        self.gamma_history = []
        
        # Register pre_hooks on all self_attn layers to apply Soft Spatial Penalty
        self.pre_hook_handles = []
        
        # Robustly find layers depending on model architecture
        try:
            layers = self.model.language_model.model.layers # HF Transformers
        except AttributeError:
            try:
                layers = self.model.model.language_model.layers # User specified alternative
            except AttributeError:
                layers = self.model.model.layers # Original LLaVA repository
                
        for layer in layers:
            handle = layer.self_attn.register_forward_pre_hook(self.pre_hook_fn, with_kwargs=True)
            self.pre_hook_handles.append(handle)
        
    def pre_hook_fn(self, module, args, kwargs):
        if not self.is_uncond_pass or self.sam_mask is None:
            return args, kwargs
            
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            kv_len = attention_mask.shape[-1]
            idx_200 = (self.guidance[0] == -200).nonzero(as_tuple=True)[0]
            if len(idx_200) > 0:
                image_idx = idx_200[0].item()
            else:
                image_token_id = getattr(self.model.config, 'image_token_index', 32000)
                idx_img = (self.guidance[0] == image_token_id).nonzero(as_tuple=True)[0]
                if len(idx_img) > 0:
                    image_idx = idx_img[0].item()
                else:
                    image_idx = 1 if self.guidance[0][0].item() in [1, 2] else 0
            
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
        # ÉP KIỂU SANG FLOAT32 để tránh lỗi 1e-9 bị làm tròn thành 0.0 trong float16 dẫn đến log(0) = -inf và 0 * -inf = NaN!
        attention_weights = uncond_attn[0, :, -1, :].to(torch.float32) # (num_heads, kv_len)
        mean_attention = attention_weights.mean(dim=0)
        
        # Tính Entropy
        H = -torch.sum(mean_attention * torch.log(mean_attention + 1e-9)).item()
        
        # Áp dụng Sigmoid
        gamma = self.gamma_scale * torch.sigmoid(torch.tensor(self.beta * (H - self.tau)))
        return gamma.item()

    def __call__(self, input_ids, logits):
        # logits là RAW logits từ mô hình chuẩn (Condition Pass)
        self.is_uncond_pass = True # Bật cờ để kích hoạt Soft Spatial Penalty
        
        if self.out is None:
            self.out = self.model(input_ids=self.guidance, 
                                  pixel_values=self.images, 
                                  attention_mask=self.attention_mask,
                                  use_cache=True,
                                  output_attentions=True)
        else:
            if self.attention_mask is not None:
                self.attention_mask = torch.cat(
                    [self.attention_mask, torch.ones((self.attention_mask.shape[0], 1), dtype=self.attention_mask.dtype, device=self.attention_mask.device)],
                    dim=1
                )
            self.out = self.model(input_ids[:, -1:],
                                  use_cache=True,
                                  attention_mask=self.attention_mask,
                                  past_key_values=self.out.past_key_values,
                                  output_attentions=True)
                                  
        self.is_uncond_pass = False # Tắt cờ

        if len(self.out.logits) == 1:
            guidance_logits = self.out.logits[0][-1:]
        else:
            guidance_logits = self.out.logits[:,-1:].to(logits.device)
            guidance_logits = guidance_logits.squeeze(1)

        # Tính Dynamic Gamma dựa trên Uncondition Attention thay vì dùng biến tĩnh
        dynamic_gamma = self.get_dynamic_gamma()
        self.gamma_history.append(dynamic_gamma)

        # Áp dụng theo lý thuyết MARINE: Khi gamma tiến tới 1 (H cao -> normal), ta tin tưởng logits (ảnh gốc).
        # Khi gamma tiến tới 0 (H thấp -> hallucinating), ta tin tưởng guidance_logits (ảnh bị che, không chứa vật thể).
        out = dynamic_gamma * logits + (1.0 - dynamic_gamma) * guidance_logits
        
        # Sửa lỗi chí mạng: Nếu dynamic_gamma == 0.0 và logits có token bị chặn (-inf), 0.0 * -inf sẽ ra NaN!
        # Tương tự nếu 1.0 - dynamic_gamma == 0.0 và guidance_logits chứa -inf.
        # Ta cần gán lại -inf cho những vị trí này để duy trì việc chặn token và tránh argmax bị sụp đổ về 0 (<unk>).
        out[logits == float('-inf')] = float('-inf')
        out[guidance_logits == float('-inf')] = float('-inf')
        
        # Đề phòng bất cứ giá trị NaN nào khác làm hỏng hàm argmax
        out = torch.nan_to_num(out, nan=float('-inf'))
        
        return out