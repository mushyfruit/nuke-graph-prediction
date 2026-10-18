def compute_top_k_accuracy(predictions, labels, ks=(1, 3, 5)):
    max_k = min(max(ks), predictions.size(1))

    _, topk_preds = predictions.topk(k=max_k, dim=1, largest=True, sorted=True)

    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))

    result = {}
    for k in ks:
        correct_k = correct[:, :k].any(dim=1).float().sum().item()

        result[k] = correct_k / labels.size(0) * 100.0
    return result
