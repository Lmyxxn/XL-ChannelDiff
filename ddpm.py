import torch
import gc

class DDPM(torch.nn.Module):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__() 
        betas = torch.linspace(min_beta, max_beta, n_steps, dtype=torch.float32).to(device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.n_steps = n_steps
        
        del betas, alphas
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def sample_forward(self, h_full, t, eps=None):
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(h_full, dtype=h_full.dtype)
        
        res = torch.sqrt(alpha_bar) * h_full

        res.add_(eps * torch.sqrt(1 - alpha_bar))

        del alpha_bar
        return res

    def __del__(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        try:
            gc.collect()
        except:
            pass