import torch

def anneal_dsm_score_estimation(scorenet, samples, sigmas, index=None, anneal_power=2., hook=None):

    if index is None:
        index = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)

    used_sigmas = sigmas[index].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise

    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, index)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, index)

    return loss.mean(dim=0)
