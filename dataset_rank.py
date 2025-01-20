import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TemplateDataset(Dataset):
    def __init__(self, sql_tables, table_column_encodings, template_ids, labels, qids, masks ,weights):
        """
        Args:
            sql_tables (list of lists or np.ndarray): Each item is a binary list of table indices for a sample.
            table_column_encodings (list of np.ndarray): Each item is an array of shape [num_tables, max_num_columns, hidden_dim].
            template_ids (list of ints): Template ID for each sample, indicating which template to use.
            column_mask (np.ndarray or None): An optional mask for columns.
        """
        self.sql_tables = sql_tables  # Shape: [batch_size, num_tables]
        self.table_column_encodings = table_column_encodings  # List of [num_tables, max_num_columns, hidden_dim]
        self.template_ids = template_ids  # List of template IDs
        self.labels = labels
        self.qids = qids
        self.masks = masks
        self.weights = weights

    def __len__(self):
        return len(self.sql_tables)

    def __getitem__(self, idx):
        # Fetch a single sample
        sql_table = self.sql_tables[idx]
        table_column_encoding = self.table_column_encodings[idx]
        template_id = self.template_ids[idx]
        
        # Convert to tensor
        table_column_encoding = torch.tensor(table_column_encoding, dtype=torch.float32)  # [num_tables, max_num_columns, hidden_dim]
        
        # Table mask is directly the sql_table itself
        table_mask = torch.tensor(sql_table, dtype=torch.bool)  # [num_tables]
        label = torch.tensor(self.labels[idx])
        qid = self.qids[idx]
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        weights = torch.tensor(self.weights[idx], dtype=torch.float)

        return table_column_encoding, table_mask, template_id, qid, label, mask, weights

def collate_fn(batch):
    table_column_encodings, table_masks, template_ids, qids, labels, masks, weights = zip(*batch)
    table_column_encodings = torch.stack(table_column_encodings)
    table_masks = torch.stack(table_masks)

    labels = torch.stack(labels)
    masks = torch.stack(masks)
    weights = torch.stack(weights)
    return table_column_encodings, table_masks, template_ids,  masks, qids, labels, weights
