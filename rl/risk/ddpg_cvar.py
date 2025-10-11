import torch
from rl.risk.ddpg import DDPG
from rl.risk.cvar import empirical_var, tail_weights

class DDPG_CVaR(DDPG):
    def __init__(self, *args, alpha=0.1, **kw):
        super().__init__(*args, **kw)
        self.alpha = alpha

    def _sample_batch(self):
        batch = self.replay.sample(self.batch_size)  # adds 'ep_return' tensor
        with torch.no_grad():
            z_hat = empirical_var(batch['ep_return'], self.alpha)
            w = tail_weights(batch['ep_return'], self.alpha, z_hat)
        batch['weights'] = w
        return batch

    def train_step(self):
        B = self._sample_batch()
        # Critic
        with torch.no_grad():
            a_next = self.actor_target(B['next_obs'])
            q_next = self.critic_target(B['next_obs'], a_next)
            y = B['rew'] + self.gamma * (1 - B['done']) * q_next
        q = self.critic(B['obs'], B['act'])
        td = (q - y)
        critic_loss = (B['weights'] * td.pow(2)).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        # Actor (maximize tail-weighted Q)
        a = self.actor(B['obs'])
        q_pi = self.critic(B['obs'], a)
        actor_loss = -(B['weights'] * q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self._soft_update_targets()
