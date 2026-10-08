import torch
from tqdm import tqdm
from ddpm import DDPM
import gc

class DDIM(DDPM):
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02,
                 guidance_scale: float = 0.1):
        super().__init__(device, n_steps, min_beta, max_beta)
        self.guidance_scale = guidance_scale

        with torch.no_grad():

            sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
            sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

            self.register_buffer('sqrt_alpha_bars', sqrt_alpha_bars)
            self.register_buffer('sqrt_one_minus_alpha_bars', sqrt_one_minus_alpha_bars)
    
    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def ddim_sample_fixed(self, x, eps, ab_cur, ab_prev, eta=0.0):
        sqrt_ab_cur = torch.sqrt(ab_cur).view(-1, 1, 1, 1)
        sqrt_ab_prev = torch.sqrt(ab_prev).view(-1, 1, 1, 1)
        sqrt_one_minus_ab_cur = torch.sqrt(1 - ab_cur).view(-1, 1, 1, 1)
        sqrt_one_minus_ab_prev = torch.sqrt(1 - ab_prev).view(-1, 1, 1, 1)

        x0 = (x - sqrt_one_minus_ab_cur * eps) / sqrt_ab_cur

        if eta == 0:
            x_prev = sqrt_ab_prev * x0 + sqrt_one_minus_ab_prev * eps
        else:
            sigma_t = eta * torch.sqrt((1 - ab_prev) / (1 - ab_cur)) * torch.sqrt(1 - ab_cur / ab_prev)
            sigma_t = sigma_t.view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            x_prev = sqrt_ab_prev * x0 + torch.sqrt(1 - ab_prev - sigma_t ** 2).view(-1, 1, 1, 1) * eps + sigma_t * noise

        del sqrt_ab_cur, sqrt_ab_prev, sqrt_one_minus_ab_cur, sqrt_one_minus_ab_prev, x0
        if eta != 0:
            del sigma_t, noise
        return x_prev

    def sample_backward(self,
                       img_or_shape,
                       partial_img,
                       net,
                       discriminator,
                       mask1,
                       mask,
                       mask_ratio,
                       sample_mask_ratio_threshold,
                       device,
                       simple_var=False,
                       ddim_step=20,
                       eta=0,
                       use_guidance=True):
        if simple_var or eta != 0:
            raise ValueError('The manuscript DDIM update uses simple_var=False and eta=0.')
        if not 1 <= ddim_step <= self.n_steps:
            raise ValueError('ddim_step must be between 1 and the diffusion step count.')
        ts = torch.linspace(self.n_steps, 0,
                           (ddim_step + 1)).to(device).to(torch.long)
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape.clone().detach()
        else:
            x = torch.randn(img_or_shape).to(device)
        partial_img = partial_img.to(device)
        batch_size = x.shape[0]
        net = net.to(device)
        discriminator = discriminator.to(device)
        
        for i in tqdm(range(1, ddim_step + 1),
                     f'DDIM sampling with eta {eta} simple_var {simple_var} guidance {use_guidance}'):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alpha_bars[cur_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device, dtype=self.alpha_bars.dtype)

            t_tensor = torch.tensor([cur_t] * batch_size,
                                  dtype=torch.long).to(device)
            t_embedding = cur_t.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])

            with torch.no_grad():

                eps = net(x, t_tensor, partial_img, mask1)
                fake_h = self.sample_backward_cgan(x, t_tensor, eps)

            if use_guidance:
                with torch.enable_grad():
                    fake_h_grad = fake_h.clone().detach().requires_grad_(True)

                    if fake_h_grad.dim() == 3:
                        fake_h_grad = fake_h_grad.unsqueeze(1)

                    d_out = discriminator(fake_h_grad, t_embedding)
                    
                    d_grad = torch.autograd.grad(d_out.sum(), fake_h_grad)[0]
                    if not torch.isfinite(d_grad).all():
                        raise FloatingPointError('Nonfinite critic guidance gradient.')
                    # Chain rule through the one-step clean-channel estimate.
                    eps = eps - self.guidance_scale * torch.sqrt((1 - ab_cur) / ab_cur) * d_grad.detach()
                    del fake_h_grad, d_out, d_grad
            
            x = self.ddim_sample_fixed(x, eps, ab_cur, ab_prev, eta=0.0)

            if mask_ratio >= sample_mask_ratio_threshold and prev_t >= 0:
                # Re-noise observed entries when the missing ratio is high.

                with torch.no_grad():

                    sqrt_ab_cur = torch.sqrt(ab_prev)
                    sqrt_one_minus_ab_cur = torch.sqrt(1 - ab_prev)
                    noise = torch.randn_like(x)

                    x_known = sqrt_ab_cur * partial_img + sqrt_one_minus_ab_cur * noise
                    x = torch.where(mask.bool(), x_known, x)
                del eps, noise, x_known
            else:
                # Keep observed channel coefficients fixed.
                x = torch.where(mask.bool(), partial_img, x)
                del eps
            if i==ddim_step:
                # Restore exact observations in the final reconstruction.
                x = torch.where(mask.bool(), partial_img, x)

            del fake_h, t_tensor, t_embedding, ab_cur, ab_prev

            if not torch.isfinite(x).all():
                raise FloatingPointError('Nonfinite reconstruction during DDIM sampling.')

        del ts, partial_img
        self.clear_memory()
        return x

    def sample_backward_cgan(self,
                            img_or_shape,
                            t,
                            eps):

        batch_size = img_or_shape.shape[0]
        sqrt_alpha_bars_t = self.sqrt_alpha_bars[t].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bars_t = self.sqrt_one_minus_alpha_bars[t].view(batch_size, 1, 1, 1)

        result = img_or_shape.clone()
        result.sub_(sqrt_one_minus_alpha_bars_t * eps)
        result.div_(sqrt_alpha_bars_t)
        
        return result
