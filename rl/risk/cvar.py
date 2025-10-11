import torch

@torch.no_grad()
def empirical_var(G: torch.Tensor, alpha: float) -> torch.Tensor:
    k = max(1, int(alpha * len(G)))
    q, _ = torch.sort(G)
    return q[k-1]  # α-fraction worst return (ascending)

def cvar_ru_loss(G: torch.Tensor, alpha: float, z: torch.Tensor) -> torch.Tensor:
    # minimize: z + (1/α) E[(z - G)+]; for maximization of G
    hinge = torch.relu(z - G)
    return z + (1.0/alpha) * hinge.mean()

@torch.no_grad()
def tail_weights(G: torch.Tensor, alpha: float, z: torch.Tensor) -> torch.Tensor:
    # 1/α · 1{G ≤ z}; stabilizes to mean 1
    return (G <= z).float() / max(alpha, 1e-6)
