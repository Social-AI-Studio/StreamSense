import torch
import torch.nn.functional as F
from torch import nn
from ipdb import set_trace

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, losses, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.classification_h_loss_coef = args.classification_h_loss_coef
        self.alpha = args.alpha
        self.weight_dict = {
            'labels_loss': self.classification_h_loss_coef,
            'contrastive_loss': self.alpha
        }
        self.losses = losses
        self.margin = args.margin
        self.size_average = True
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.beta = args.beta

    def loss_labels(self, input_tuple, targets_tuple, name, dataset_name):
        input, iou_scores = input_tuple
        targets = targets_tuple
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        target = targets.float()
        log_probs = self.logsoftmax(input)  # shape [batch, 2]
        output = torch.sum(- target * (iou_scores.clamp(min=1e-6) ** self.beta) * log_probs, dim=1)
        loss_ce = torch.mean(output)

        return {name: loss_ce}

    def cosine_similarity_matrix(self, x, y):
        # Normalize each row (embedding) and compute cosine similarity
        x_norm = F.normalize(x, dim=1)  # [B, D]
        y_norm = F.normalize(y, dim=1)  # [B, D]
        return torch.matmul(x_norm, y_norm.T)  # [B, B]

    def cross_modal_contrastive_loss(self, emb1, emb2, temperature=0.07):
        sim_matrix = self.cosine_similarity_matrix(emb1, emb2)  # [B, B]
        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        
        loss_1_to_2 = F.cross_entropy(sim_matrix / temperature, labels)
        loss_2_to_1 = F.cross_entropy(sim_matrix.T / temperature, labels)
        
        return (loss_1_to_2 + loss_2_to_1) / 2
        
    def contrastive_loss(self, input, labels, name, dataset_name):
        temperature=0.07
        input_text = input.get("text")     # [B, D]
        input_audio = input.get("audio")   # [B, D]
        input_visual = input.get("visual") # [B, D]
        labels = torch.argmax(labels, dim=1)  # [B]

        contrastive_loss = 0.0
        # Cross-modal losses
        if input_text is not None and input_audio is not None:
            contrastive_loss += 0.5 * self.cross_modal_contrastive_loss(input_text, input_audio, temperature)
        if input_text is not None and input_visual is not None:
            contrastive_loss += 0.5 * self.cross_modal_contrastive_loss(input_text, input_visual, temperature)
        
        return {name: contrastive_loss}

    def get_loss(self, loss, dataset_name, outputs, targets):
        loss_map = {
            'labels_loss': self.loss_labels,
            'contrastive_loss': self.contrastive_loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, name=loss, dataset_name=dataset_name)

    def forward(self, outputs, targets, dataset_name):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, dataset_name, outputs[loss], targets[loss]))

        return losses