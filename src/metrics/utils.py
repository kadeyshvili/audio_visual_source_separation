import torch 


def calc_si_sdr(est, target, eps=1e-10):
    alpha = (est * target).sum() / ( torch.square(target).sum() + eps)
    return 10 * torch.log10(torch.square(alpha * target).sum() / (torch.square(alpha * target - est).sum() + eps))