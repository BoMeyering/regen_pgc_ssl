# Welfords method for calculating mean and variance
# BoMeyering 2024
import torch

class WelfordCalculator:
    def __init__(self):
        self.M = torch.zeros(3)
        self.S = torch.zeros(3)
        self.N = 0

    def _update(self, array: torch.Tensor):
        if len(array.shape) > 3:
            raise ValueError(f"Passed tensor should have shape of len=3. Current 'array' argument is of shape {tensor.shape}")
        elif array.shape[0] != 3:
            raise ValueError(f"Argument 'array' should be a 3 channel image. Tensor passed has {tensor.shape[0]} channels.")
        num_px = array.shape[1] * array.shape[2]
        self.N += num_px
        old_M = self.M.clone()
        M_new = array.mean(dim=(1, 2))
        
        D1 = M_new - self.M
        self.M += D1 * num_px / self.N
        # print("M: ", self.M)

        # S_term = (array - M_new.unsqueeze(-1).unsqueeze(-1)) * (array - self.M.unsqueeze(-1).unsqueeze(-1))
        # self.S += S_term.sum(dim=(1, 2))
        
        array_reshaped = array.reshape(3, -1)  # Reshape to (3, H*W)
        old_M_expanded = old_M.unsqueeze(-1).expand_as(array_reshaped)
        M_new_expanded = M_new.unsqueeze(-1).expand_as(array_reshaped)
        self.S += ((array_reshaped - old_M_expanded) * (array_reshaped - M_new_expanded)).sum(dim=1)

        # print("New Terms", self.M, self.S)

    def compute(self):
        variance = self.S / self.N
        std = torch.sqrt(variance)

        return self.M, std

if __name__ == '__main__':
    welford = WelfordCalculator()

    means = torch.zeros(3)
    std = torch.zeros(3)
    pixel_n = torch.zeros(1)

    for i in range(100000):
        tensor = torch.randn(3, 10, 10)+10
        welford._update(tensor)

        channel_means = torch.sum(tensor, dim=(1, 2))
        channel_std = torch.sum(tensor** 2, dim=(1, 2))
        pixels = tensor.shape[2] * tensor.shape[1]

        means += channel_means
        std += channel_std
        pixel_n += pixels

        if i % 1000 == 0:
            mean, variance = welford.compute()
            print(mean,variance)

            print(means / pixel_n, channel_std * pixel_n / i)