import torch
import random
import numpy as np
from math import radians, cos, sin, asin, sqrt


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)

    return scale * tensor / (torch.sqrt(squared_norm) + 1e-8)


def mean_reciprocal_rank(prob, targets):

    # prob = torch.softmax(prob, dim=-1)
    _, indices = torch.sort(prob, dim=1, descending=True)

    sorted_labels = torch.gather(targets, 1, indices)

    rank = (sorted_labels == 1).nonzero(as_tuple=True)[1] + 1

    reciprocal_rank = 1.0 / rank.float()

    mrr = torch.sum(reciprocal_rank)

    return mrr.item()


def calculate_true_positive(prob, target):

    acc_set = torch.zeros((4,), dtype=torch.long)

    _, target = torch.topk(target, k=1)

    for i, k in enumerate([1, 5, 10, 20]):

        _, predict_top = torch.topk(prob, k=k)

        for j, pt in enumerate(predict_top):

            if target[j] in pt:
                acc_set[i] += 1

    return np.array(acc_set)


def regenerate_prob(prob, target, num_neg):

    # prob:(bs, num_poi) target:(bs, num_poi)

    bs, num_poi = prob.shape[0], prob.shape[1]
    _, target = torch.topk(target, k=1)  # (bs, 1)
    re_target = torch.linspace(0, bs-1, bs, dtype=torch.long)  # [0 ~ bs-1] every element is 0 or 1
    re_prob = torch.zeros((bs, num_neg+len(target)), dtype=torch.float)

    neg_random = random.sample(range(0, num_poi), num_neg)  # (num_neg) from (0 -- num_poi-1) because poi consider the 0
    while len([lab for lab in target if lab in neg_random]) != 0:  # no intersection
        neg_random = random.sample(range(0, num_poi), num_neg)

    # place the pos labels ahead and neg samples in the end
    for b in range(bs):  # in range(bs)
        for t in range(num_neg + len(target)):  # in range(the final length)
            if t < len(target):
                re_prob[b, t] = prob[b, target[t]]  # get the prob of targets
            else:
                re_prob[b, t] = prob[b, neg_random[t - len(target)]]  # get the prob of negative targets

    return torch.Tensor(re_prob), torch.Tensor(re_target)  # (bs, num_neg+num_target), (bs)


def timestamps_to_hours(timestamps):
    seconds_in_hour = 3600

    hours_since_epoch = timestamps // seconds_in_hour
    hours = hours_since_epoch % 24

    return hours


def haversine(lat1, lon1, lat2, lon2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r
