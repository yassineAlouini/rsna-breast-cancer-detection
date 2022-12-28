# The evaluation metric but probably not the best loss...
# Source: https://aclanthology.org/2020.eval4nlp-1.9.pdf

def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

# From: https://www.kaggle.com/code/tanreinama/training-efficientnet-with-tpu-in-rsna-screening
def pfbeta_torch(labels, predictions, beta=1.0):
    y_true_count = torch.sum(labels)
    ctp = 0
    cfp = 0

    predictions = torch.clamp(predictions, min=0, max=1)
    ctp = torch.sum(predictions * labels)
    cfp = torch.sum(predictions * (1.0-labels))

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0