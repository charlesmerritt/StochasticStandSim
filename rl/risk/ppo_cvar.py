import torch
from stable_baselines3 import PPO
from rl.risk.cvar import empirical_var, cvar_ru_loss, tail_weights

class PPO_CVaR(PPO):
    def __init__(self, *args, alpha=0.1, lambda_z=1.0, **kw):
        super().__init__(*args, **kw)
        self.alpha = alpha
        self.z = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.lambda_z = lambda_z
        self.ru_optim = torch.optim.Adam([self.z], lr=3e-4)

    def compute_policy_loss(self, mb_obs, mb_actions, mb_logp_old, mb_adv, mb_returns, mb_ep_returns):
        # mb_ep_returns: per-sample episode return G_i, aligned with advantages
        with torch.no_grad():
            z_hat = empirical_var(mb_ep_returns, self.alpha)
            w = tail_weights(mb_ep_returns, self.alpha, z_hat)

        dist = self.actor(mb_obs)
        logp = dist.log_prob(mb_actions)
        ratio = torch.exp(logp - mb_logp_old)

        adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
        adv_cvar = w * adv  # emphasize tail

        unclipped = ratio * adv_cvar
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_cvar
        pi_loss = -torch.mean(torch.min(unclipped, clipped))

        # RU step for z on the same minibatch of episode returns
        ru_loss = cvar_ru_loss(mb_ep_returns.detach(), self.alpha, self.z)
        (self.lambda_z * ru_loss).backward(retain_graph=False)
        self.ru_optim.step(); self.ru_optim.zero_grad(set_to_none=True)

        return pi_loss

    # value loss unchanged or add a CVaR-critic head later
