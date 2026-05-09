import torch.utils.data
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

class RankAccuracy(Metric):

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        self._num_correct = None
        self._num_examples = None
        super(RankAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RankAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        (rank_left, rank_right, label) = output
        rank_left = rank_left.view(-1)
        rank_right = rank_right.view(-1)
        label = label.view(-1)

        # Compute accuracy for non ties
        index_mask = label != 0                                 # Select non-ties
        aux_label = -1 * label[index_mask]                      # Invert label (-1 for right winning, 1 for left)
        diff = rank_left[index_mask] - rank_right[index_mask]   # Compute rank difference
        '''
        # ---- DEBUG BEGIN ----
        if aux_label.numel() > 0:
            
            # simple sign-based correctness (what RankAccuracy uses)
            correct = ((aux_label == 1) & (diff > 0)) | ((aux_label == -1) & (diff < 0))
            batch_acc = correct.float().mean().item()
    
            frac_zero = (diff == 0).float().mean().item()
            mean_diff = diff.mean().item()
            std_diff = diff.std().item() if diff.numel() > 1 else 0.0
     
                print(
                f"[DEBUG RankAccuracy] "
                f"batch_non_ties={aux_label.numel()} "
                f"batch_acc={batch_acc:.4f} "
                f"mean_diff={mean_diff:.4f} std_diff={std_diff:.4f} "
                f"frac_diff_zero={frac_zero:.4f}"
            )
            
        
        else:
            print("[DEBUG RankAccuracy] batch has NO non-ties")
        # ---- DEBUG END ----
        '''
        correct_left = (aux_label == 1) & (diff > 0)            # Compute correct rank for left choices
        correct_right = (aux_label == -1) & (diff < 0)          # Compute correct rank for right choices
        self._num_correct += torch.sum(correct_left + correct_right).item()
        self._num_examples += aux_label.size()[0]



    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples


class RankAccuracy_withMargin(Metric):
    def __init__(self, output_transform=lambda x: x, device='cpu'):
        self._num_correct = None
        self._num_examples = None
        super(RankAccuracy_withMargin, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RankAccuracy_withMargin, self).reset()

    @reinit__is_reduced
    def update(self, output):
        (rank_left, rank_right, label, margin) = output
        rank_left = rank_left.view(-1)
        rank_right = rank_right.view(-1)
        label = label.view(-1)

        # Compute accuracy for non ties
        index_mask = label != 0                                   # Select non-ties
        aux_label = -1 * label[index_mask]                        # Invert label (-1 for right winning, 1 for left)
        diff = rank_left[index_mask] - rank_right[index_mask]     # Compute rank difference
        correct_left = (aux_label == 1) & (diff > margin)         # Compute correct rank for left choices
        correct_right = (aux_label == -1) & (diff < -1 * margin)  # Compute correct rank for right choices

        self._num_correct += torch.sum(correct_left + correct_right).item()
        self._num_examples += aux_label.size()[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples

class RankAccuracy_ties(Metric):

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        self._num_correct = None
        self._num_examples = None
        super(RankAccuracy_ties, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RankAccuracy_ties, self).reset()

    @reinit__is_reduced
    def update(self, output):
        (rank_left, rank_right, label, margin) = output
        rank_left = rank_left.view(-1)
        rank_right = rank_right.view(-1)
        label = label.view(-1)

        index_mask = label == 0                                       # Select ties
        aux_label = label[index_mask]                                 # Get labels
        diff = rank_left[index_mask] - rank_right[index_mask]         # Compute rank difference
        correct_ties = (aux_label == 0) & (torch.abs(diff) < margin)  # Compute correct rank for left choices
        self._num_correct += torch.sum(correct_ties).item()
        self._num_examples += aux_label.size()[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples


class ClassificationAUC(Metric):
    """
    Epoch-level one-vs-rest ROC AUC for classification logits.

    Binary runs report the positive-class AUC. Multiclass runs report a weighted
    mean of per-class one-vs-rest AUC values, skipping classes that are absent or
    have no negatives in the current split.
    """

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        super(ClassificationAUC, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._scores = []
        self._targets = []
        super(ClassificationAUC, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        if y_pred is None or y is None:
            return

        scores = torch.softmax(y_pred.detach(), dim=1).cpu()
        targets = y.detach().long().view(-1).cpu()

        if scores.ndim != 2:
            raise TypeError(f"ClassificationAUC expected logits [B,C], got shape {tuple(scores.shape)}")
        if targets.numel() != scores.size(0):
            raise TypeError(
                f"ClassificationAUC target count {targets.numel()} does not match logits batch {scores.size(0)}"
            )

        self._scores.append(scores)
        self._targets.append(targets)

    def compute(self):
        if not self._scores:
            return float("nan")

        scores = torch.cat(self._scores, dim=0)
        targets = torch.cat(self._targets, dim=0)

        n_classes = int(scores.size(1))
        if n_classes < 2:
            return float("nan")

        y_true = targets.numpy()
        y_score = scores.numpy()

        if n_classes == 2:
            if len(set(y_true.tolist())) < 2:
                return float("nan")
            return float(roc_auc_score(y_true, y_score[:, 1]))

        auc_sum = 0.0
        weight_sum = 0.0
        for cls_idx in range(n_classes):
            binary_true = (targets == cls_idx).numpy().astype(int)
            positives = int(binary_true.sum())
            negatives = int(binary_true.shape[0] - positives)
            if positives == 0 or negatives == 0:
                continue

            auc_sum += float(roc_auc_score(binary_true, y_score[:, cls_idx])) * positives
            weight_sum += float(positives)

        if weight_sum <= 0.0:
            return float("nan")
        return auc_sum / weight_sum


class RankAUC(Metric):
    """
    ROC AUC for the pairwise ranking branch.

    Uses non-tie pairs only. The positive class is "left is preferred" and the
    score is rank_left - rank_right, so larger values should favor the left item.
    """

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        super(RankAUC, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._scores = []
        self._targets = []
        super(RankAUC, self).reset()

    @reinit__is_reduced
    def update(self, output):
        rank_left, rank_right, label = output
        rank_left = rank_left.detach().view(-1).cpu()
        rank_right = rank_right.detach().view(-1).cpu()
        label = label.detach().view(-1).cpu()

        non_tie_mask = label != 0
        if non_tie_mask.sum().item() == 0:
            return

        diff = rank_left[non_tie_mask] - rank_right[non_tie_mask]
        target_left_preferred = (label[non_tie_mask] == -1).long()

        self._scores.append(diff)
        self._targets.append(target_left_preferred)

    def compute(self):
        if not self._scores:
            return float("nan")

        scores = torch.cat(self._scores, dim=0).numpy()
        targets = torch.cat(self._targets, dim=0).numpy()

        if len(set(targets.tolist())) < 2:
            return float("nan")

        return float(roc_auc_score(targets, scores))


class ClassificationSensitivity(Metric):
    """
    Epoch-level sensitivity/recall for classification logits.

    Binary classification reports recall for class 1. Multiclass classification
    reports weighted recall so tie-enabled runs still get a single scalar.
    """

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        super(ClassificationSensitivity, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._preds = []
        self._targets = []
        super(ClassificationSensitivity, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        if y_pred is None or y is None:
            return

        preds = torch.argmax(y_pred.detach(), dim=1).view(-1).cpu()
        targets = y.detach().long().view(-1).cpu()

        if targets.numel() != preds.numel():
            raise TypeError(
                f"ClassificationSensitivity target count {targets.numel()} does not match predictions {preds.numel()}"
            )

        self._preds.append(preds)
        self._targets.append(targets)

    def compute(self):
        if not self._preds:
            return float("nan")

        preds = torch.cat(self._preds, dim=0).numpy()
        targets = torch.cat(self._targets, dim=0).numpy()
        average = "binary" if len(set(targets.tolist())) <= 2 and len(set(preds.tolist() + targets.tolist())) <= 2 else "weighted"
        return float(recall_score(targets, preds, average=average, zero_division=0))


class ClassificationF1(Metric):
    """
    Epoch-level F1 score for classification logits.

    Binary classification reports F1 for class 1. Multiclass classification
    reports weighted F1 so tie-enabled runs still get a single scalar.
    """

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        super(ClassificationF1, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._preds = []
        self._targets = []
        super(ClassificationF1, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        if y_pred is None or y is None:
            return

        preds = torch.argmax(y_pred.detach(), dim=1).view(-1).cpu()
        targets = y.detach().long().view(-1).cpu()

        if targets.numel() != preds.numel():
            raise TypeError(
                f"ClassificationF1 target count {targets.numel()} does not match predictions {preds.numel()}"
            )

        self._preds.append(preds)
        self._targets.append(targets)

    def compute(self):
        if not self._preds:
            return float("nan")

        preds = torch.cat(self._preds, dim=0).numpy()
        targets = torch.cat(self._targets, dim=0).numpy()
        average = "binary" if len(set(targets.tolist())) <= 2 and len(set(preds.tolist() + targets.tolist())) <= 2 else "weighted"
        return float(f1_score(targets, preds, average=average, zero_division=0))


class RankSensitivity(Metric):
    """
    Sensitivity/recall for the pairwise ranking branch.

    Uses non-tie pairs only. The positive class is "left is preferred" and the
    prediction is rank_left > rank_right.
    """

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        super(RankSensitivity, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._preds = []
        self._targets = []
        super(RankSensitivity, self).reset()

    @reinit__is_reduced
    def update(self, output):
        rank_left, rank_right, label = output
        rank_left = rank_left.detach().view(-1).cpu()
        rank_right = rank_right.detach().view(-1).cpu()
        label = label.detach().view(-1).cpu()

        non_tie_mask = label != 0
        if non_tie_mask.sum().item() == 0:
            return

        preds = (rank_left[non_tie_mask] > rank_right[non_tie_mask]).long()
        targets = (label[non_tie_mask] == -1).long()
        self._preds.append(preds)
        self._targets.append(targets)

    def compute(self):
        if not self._preds:
            return float("nan")

        preds = torch.cat(self._preds, dim=0).numpy()
        targets = torch.cat(self._targets, dim=0).numpy()
        return float(recall_score(targets, preds, zero_division=0))


class RankF1(Metric):
    """
    F1 score for the pairwise ranking branch.

    Uses non-tie pairs only. The positive class is "left is preferred" and the
    prediction is rank_left > rank_right.
    """

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        super(RankF1, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._preds = []
        self._targets = []
        super(RankF1, self).reset()

    @reinit__is_reduced
    def update(self, output):
        rank_left, rank_right, label = output
        rank_left = rank_left.detach().view(-1).cpu()
        rank_right = rank_right.detach().view(-1).cpu()
        label = label.detach().view(-1).cpu()

        non_tie_mask = label != 0
        if non_tie_mask.sum().item() == 0:
            return

        preds = (rank_left[non_tie_mask] > rank_right[non_tie_mask]).long()
        targets = (label[non_tie_mask] == -1).long()
        self._preds.append(preds)
        self._targets.append(targets)

    def compute(self):
        if not self._preds:
            return float("nan")

        preds = torch.cat(self._preds, dim=0).numpy()
        targets = torch.cat(self._targets, dim=0).numpy()
        return float(f1_score(targets, preds, zero_division=0))


if __name__ == '__main__':
    import torch
    torch.manual_seed(8)

    m = RankAccuracy()
    output = (
        torch.Tensor([1, 2, 3, 1]),
        torch.Tensor([0, 3, 1, 3]),
        torch.Tensor([1, -1, 0, 1])
    )

    m.update(output)
    m.update(output)
    res = m.compute()

    print(m._num_correct, m._num_examples, res)
