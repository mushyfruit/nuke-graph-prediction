def compute_top_k_accuracy(predictions, labels, ks=(1, 3, 5)):
    max_k = max(ks)

    # Get indicies of topk predictions.
    _, topk_preds = predictions.topk(k=max_k, dim=1, largest=True, sorted=True)

    # Compute element-wise equality between topk and ground-truth.
    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))

    result = {}
    for k in ks:
        # Evaluate how many samples were correct at a given k value.
        correct_k = correct[:, :k].any(dim=1).float().sum().item()

        result[k] = correct_k / labels.size(0) * 100.0
    return result
