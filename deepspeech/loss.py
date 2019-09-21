import torch

class Loss(object):
    """A simple wrapper class for loss calculation"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CTCLoss(reduction='sum').to(self.device)

    def calculate_loss(self, inputs, input_sizes, targets, target_sizes):
        """Calculate CTC loss.
        Args:
            logits: N x T x C, score before softmax
            logits_sizes: number of timesteps of logits 
            targets: N x T
            target_sizes: number of timesteps of targets
        """
        inputs = inputs.to(self.device)
        input_sizes = input_sizes.to(self.device)
        targets = targets.to(self.device)
        target_sizes = target_sizes.to(self.device)

        logits, logit_sizes = self.model(inputs, input_sizes)
        out = logits.transpose(0, 1)  # TxNxC
        out = out.log_softmax(-1)
    
        out = out.float()  # ensure float32 for loss
        loss = self.criterion(out, targets, logit_sizes, target_sizes)
        loss = loss / logits.size(0)  # average the loss by minibatch

        if loss.item() == float("inf") or loss.item() == float("-inf"):
            raise Exception("WARNING: received an inf loss")
        if torch.isnan(loss).sum() > 0:
            raise Exception('WARNING: received a nan loss')
        if loss.item() < 0:
            raise Exception("WARNING: received a negative loss")
        return loss