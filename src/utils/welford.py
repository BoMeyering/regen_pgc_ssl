# Welfords method for calculating mean and variance
# BoMeyering 2024
import torch
import sys

class WelfordCalculator:
    def __init__(self):
        self.M = torch.zeros(3, dtype=torch.float64)
        self.M_old = torch.zeros(3, dtype=torch.float64)
        self.S = torch.zeros(3, dtype=torch.float64)
        self.N = torch.zeros(1, dtype=torch.long)

    def update(self, array: torch.Tensor):
        if (len(array.shape) > 3) or (array.shape[0] != 3):
            raise ValueError(f"Passed tensor should be a 3 channel image of shape (3, H, W). Current 'array' argument is of shape {array.shape}")
        
        # Update count based on number of pixels
        num_px = array.shape[1] * array.shape[2]
        self.N += num_px

        # Calculate array mean and update running mean
        self.M_old = self.M.clone()
        m_new = array.mean(dim=(1, 2))
        self.M += (m_new - self.M) * num_px / self.N

        # Calculate new S incremental term (x-M_t) * (x-M_{t-1})
        S_incr = (array - self.M.unsqueeze(-1).unsqueeze(-1)) * (array - self.M_old.unsqueeze(-1).unsqueeze(-1))
        self.S += S_incr.sum(dim=(1, 2))

    def compute(self):
        self.std = torch.sqrt(self.S / self.N)

        return self.M, self.std
