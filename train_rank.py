import re
import torch, logging
import torch.nn as nn
import torch.optim as optim
from typing import List
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict
logger = logging.getLogger('mylogger')

def train_model(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device,
                 num_epochs: int = 10, cur_epoch=None, scheduler=None):
    """
    Train the model.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch in dataloader:
            # Move data to the device
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            weights = batch[-1]
            y_true = batch[-2]
            masks = batch[-4]

            optimizer.zero_grad()
            y_pred = model(*batch[:-4])
            
            # Expand y_pred to shape (batch_size, num_candidates, num_candidates)
            y_pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # (batch_size, num_candidates, num_candidates)

            # Calculate the true label differences (1 if y_true[i] < y_true[j], otherwise 0)
            y_true_diff = (y_true.unsqueeze(2) < y_true.unsqueeze(1)).float()  # (batch_size, num_candidates, num_candidates)

            # Broadcast mask to calculate valid candidate pairs
            mask_diff = masks.unsqueeze(2) * masks.unsqueeze(1)  # (batch_size, num_candidates, num_candidates)
            num_candidates = mask_diff.size(1)
            mask_diff[:, torch.arange(num_candidates), torch.arange(num_candidates)] = 0 

            # Use BCEWithLogitsLoss to calculate the loss for each pair of candidates
            loss = criterion(y_pred_diff, y_true_diff)  # Compute loss for each pair
            
            # Set the loss for invalid candidate pairs to 0
            loss = loss * mask_diff  # (batch_size, num_candidates, num_candidates)

            # Sum the loss for all candidate pairs and compute the average loss
            # Need to compute the average loss based on the number of valid pairs
            valid_pairs = mask_diff.sum()  # Count valid candidate pairs
            if valid_pairs > 0:
                loss = (loss * weights).sum() / valid_pairs
            else:
                loss = torch.tensor(0.0, device=y_pred.device)
                
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        avg_loss = epoch_loss / len(dataloader)
        if (cur_epoch+1) % 1 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}", extra={'end':''})
        if scheduler is not None:
            scheduler.step()
    return avg_loss



def test_model(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device, valid_dataloader=None):
    """
    Evaluate the model.
    """
    model.eval()
    total_loss = 0
    latency_sum = 0

    # Initialize template accuracy dictionary
    template_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    template_latency = defaultdict(float)
    best_correct = 0
    total = 0
    best_true_latency = 0
    for batch in dataloader:
        # Move data to the device
        batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        weights = batch[-1]
        y_true = batch[-2]
        masks = batch[-4]
        template_ids = batch[-5]

        y_pred =  model(*batch[:-4])
        batch_size, num_candidates = y_pred.size()
        best_true_latency += torch.min(y_true, dim=1)[0].sum()
        
        # Calculate pairwise accuracy
        pred_order = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # (batch_size, num_candidates, num_candidates)
        true_order = y_true.unsqueeze(2) <= y_true.unsqueeze(1)  # (batch_size, num_candidates, num_candidates)
        
        mask_pairs = masks.unsqueeze(2) * masks.unsqueeze(1)
        mask_diag = torch.eye(num_candidates, device=masks.device).unsqueeze(0).expand(batch_size, -1, -1)
        mask_pairs = mask_pairs * (1 - mask_diag.float())  # Only set diagonal elements of valid pairs to zero

        sign_pred = pred_order >= 0
        pairwise_accuracy = (true_order == sign_pred) * mask_pairs.float()

        # Calculate the number of valid candidate pairs
        valid_pairs_count = mask_pairs.sum()
        
        # Compute accuracy, denominator is the number of valid candidate pairs
        pairwise_accuracy = pairwise_accuracy.sum() / valid_pairs_count
        
        # Compute the sum of true values for the best predicted candidates
        masked_y_pred = y_pred.masked_fill(masks == 0, float('-inf'))  # Set predicted values to inf for invalid candidates
        _, best_pred_idx = masked_y_pred.max(dim=1)  # Index of the best predicted candidate (batch_size,)


        best_true_sum = torch.gather(y_true, 1, best_pred_idx.unsqueeze(1))  # Gather true values for the best predicted candidates (batch_size, 1)
        best_true_sum = best_true_sum.sum()  # Sum across all samples
        latency_sum += best_true_sum

        # Compute the loss for each pair
        loss = criterion(pred_order, true_order.float())  # Compute loss for each pair

        # Set the loss for invalid candidate pairs to 0
        loss = loss * mask_pairs  # (batch_size, num_candidates, num_candidates)

        _, true_best_idx = y_true.min(dim=1)
        best_correct += (best_pred_idx == true_best_idx).sum()
        total += len(true_best_idx)

        # Sum the loss for all candidate pairs and compute the average loss
        valid_pairs = mask_pairs.sum()  # Count valid candidate pairs
        if valid_pairs > 0:
            loss = (loss * weights).sum() / valid_pairs
        else:
            loss = torch.tensor(0.0, device=y_pred.device)
        total_loss += loss

        # Track accuracy for each template
        for i in range(batch_size):
            template_id = template_ids[i]  # Template ID for the current sample
            pred_label = best_pred_idx[i].item()  # The best candidate predicted for the current sample
            true_label = torch.argmin(y_true[i]).item()  # The true best candidate label for the current sample (the one with the minimum value)
            
            if pred_label == true_label:
                template_accuracy[template_id]['correct'] += 1
            template_accuracy[template_id]['total'] += 1
            template_latency[template_id] += y_true[i][pred_label]

    avg_loss = total_loss / len(dataloader)
    template_accuracies = {id: stats['correct'] / stats['total'] for id, stats in template_accuracy.items()}
    best_accuracy = best_correct / total
    

    valid_latency_sum = 0
    if valid_dataloader:
        for batch in valid_dataloader:
            # Move data to the device
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            y_true = batch[-2]
            masks = batch[-4]

            y_pred =  model(*batch[:-4])
            batch_size, num_candidates = y_pred.size()
     
            
            # Compute the sum of true values for the best predicted candidates
            masked_y_pred = y_pred.masked_fill(masks == 0, float('-inf'))  # Set predicted values to inf for invalid candidates
            _, best_pred_idx = masked_y_pred.max(dim=1)  # Index of the best predicted candidate (batch_size,)


            best_true_sum = torch.gather(y_true, 1, best_pred_idx.unsqueeze(1))  # Gather true values for the best predicted candidates (batch_size, 1)
            best_true_sum = best_true_sum.sum()  # Sum across all samples
            valid_latency_sum += best_true_sum


    logger.info(f"Validation Loss: {avg_loss:.4f}, Acc: {pairwise_accuracy:.3f}, Latency: {latency_sum:.2f}, best_true_latency: {best_true_latency:.2f}, valid_latency: {valid_latency_sum:.2f}")

    model.train()
    return avg_loss, pairwise_accuracy, latency_sum, template_accuracies, best_accuracy, template_latency, best_true_latency, valid_latency_sum

def predict(model: nn.Module, dataloader: DataLoader,  device: torch.device):
    for batch in dataloader:
        masks = batch[3].to(device)
        input = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch[:3]]
        y_pred =  model(*input)
        masked_y_pred = y_pred.masked_fill(masks == 0, float('-inf'))
        _, best_pred_idx = masked_y_pred.max(dim=1)
        break
    return best_pred_idx
