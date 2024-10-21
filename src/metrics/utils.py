import torch 


def calc_si_sdr(est, target):
    alpha = (est* target).sum() / ( torch.square(target).sum() )
    return 10*torch.log10(torch.square(alpha * target).sum() / torch.square(alpha * target - est).sum())