import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgan_enhanced_ddim import DDIM
from DiT.blocks import MaskedMultiHeadCrossAttention
from GAN.models_enhanced_wcgan import Discriminator


class ZeroNet(torch.nn.Module):
    def forward(self, x, t, partial, mask):
        return torch.zeros_like(x)


class LinearCritic(torch.nn.Module):
    def forward(self, x, t=None):
        return x.flatten(1).sum(1, keepdim=True)


class PaperAlignmentTests(unittest.TestCase):
    def test_noise_guidance_increases_clean_channel_score(self):
        sampler = DDIM('cpu', 1000)
        a = sampler.alpha_bars[500]
        x, eps = torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4)
        grad = torch.ones_like(x)
        guided = eps - .1 * ((1 - a) / a).sqrt() * grad
        before = sampler.sample_backward_cgan(x, torch.full((2,), 500), eps)
        after = sampler.sample_backward_cgan(x, torch.full((2,), 500), guided)
        self.assertTrue(torch.all(after > before))

    def test_reverse_update_and_final_observations(self):
        sampler = DDIM('cpu', 1000)
        mask = torch.ones(2, 4, 4)
        mask[:, 2:] = 0
        partial = torch.randn(2, 2, 4, 4) * mask[:, None]
        for ratio in [.5, .7, .8]:
            outputs = []
            for _ in range(2):
                torch.manual_seed(42)
                out = sampler.sample_backward(partial.shape, partial, ZeroNet(), LinearCritic(),
                                              mask, mask[:, None], ratio, .7, 'cpu', ddim_step=5)
                self.assertTrue(torch.equal(out * mask[:, None], partial))
                self.assertTrue(torch.isfinite(out).all())
                outputs.append(out)
            self.assertTrue(torch.equal(*outputs))
        with self.assertRaises(ValueError):
            sampler.sample_backward(partial.shape, partial, ZeroNet(), LinearCritic(),
                                    mask, mask[:, None], .5, .7, 'cpu', eta=1)

    def test_unknown_queries_ignore_unknown_keys(self):
        torch.manual_seed(42)
        attention = MaskedMultiHeadCrossAttention(16, 4).eval()
        x, cond = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        mask = torch.tensor([[[1, 1, 0, 0]], [[1, 0, 1, 0]]])
        changed = cond.clone()
        changed[~mask.squeeze(1).bool()] += 1000
        self.assertTrue(torch.allclose(attention(x, cond, mask), attention(x, changed, mask)))

    def test_adversarial_gradient_reaches_generator(self):
        critic = Discriminator(2).eval()
        fake = torch.randn(2, 2, 32, 32, requires_grad=True)
        grad = torch.autograd.grad(-critic(fake).mean(), fake)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        with torch.no_grad():
            for p in critic.parameters():
                p.clamp_(-.01, .01)
        self.assertTrue(all(float(p.detach().abs().max()) <= .010001 for p in critic.parameters()))


if __name__ == '__main__':
    unittest.main()
