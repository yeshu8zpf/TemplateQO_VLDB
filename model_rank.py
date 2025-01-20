from copy import deepcopy
import torch
import torch.nn as nn


import torch
import torch.nn as nn



class TemplateAwarePlanSelector2(nn.Module):
    def __init__(self, col_encoding_leng, hidden_dim, n_tables, template_ids, num_heads, num_layers, max_num_column, num_classes, dropout=0., 
                 table_index_encoding_len=10, filter_col_types=[]):
        super(TemplateAwarePlanSelector2, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_tables = n_tables
        self.n_templates = len(template_ids)
        self.col_encoding_leng = col_encoding_leng
        self.table_index_encoding_len = table_index_encoding_len
        self.filter_col_types = filter_col_types
        self.template_ids = template_ids
        self.id2index = {id: i for i, id in enumerate(template_ids)}

        transformer_encoder_layer = nn.TransformerEncoderLayer(nhead=num_heads, dim_feedforward=hidden_dim, d_model=hidden_dim,
                                                               dropout=dropout, batch_first=True)
        self.template_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        
        # Initialize template_queries for each template
        self.template_queries = nn.Parameter(torch.randn(self.n_templates, hidden_dim))
        self.table_index_encoding = nn.Parameter(torch.randn(n_tables, table_index_encoding_len), requires_grad=False)
        self.template_emb = nn.Parameter(torch.randn(self.n_templates, table_index_encoding_len), requires_grad=False)

        # Initialize classification heads for each template, where each template has its own number of classes
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim+table_index_encoding_len, hidden_dim//2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)) 
        
        self.filter_projects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(col_encoding_leng*max_num_column, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(2*hidden_dim, hidden_dim-table_index_encoding_len),
                nn.ReLU(),
                ) for _ in range(n_tables)
        ])
        self.filter_projects, self.filter_masks = self.bulid_filter_project_masks(filter_col_types)

    def bulid_filter_project_masks(self, filter_column_types):
        masks = []
        projects = []
        for types in filter_column_types:
            mask = []
            for i, type in enumerate(types):
                if type == 0:
                    mask.extend(range(self.col_encoding_leng*i, self.col_encoding_leng*i+3))
                else:
                    mask.extend(range(self.col_encoding_leng*i, self.col_encoding_leng*(i+1)))
            masks.append(torch.tensor(mask))
            projects.append( nn.Sequential(
                nn.Linear(len(mask), self.hidden_dim*2),
                nn.ReLU(),
                nn.Linear(2*self.hidden_dim, self.hidden_dim-self.table_index_encoding_len),
                nn.ReLU(),
                ))
        return nn.ModuleList(projects), masks

    def forward(self, table_column_encodings, table_masks, template_ids):
        # table_column_encodings: [batch_size, n_tables, n_columns, hidden_dim]
        # table_masks: [batch_size, n_tables] (this is per sample)
        # template_ids: [batch_size] (this is per sample)

        batch_size, n_tables, n_columns, col_dim = table_column_encodings.shape
        
        table_encodings = []
        for i in range(n_tables):

            table_col_encoding = table_column_encodings[:, i, :, :].reshape(batch_size, -1)[:, self.filter_masks[i]]
            table_encoding = self.filter_projects[i](table_col_encoding)  # Shape: [batch_size, hidden_dim]
            # Append to the list of table_encodings
            table_encodings.append(table_encoding)
        
        # Stack all table encodings into one tensor
        table_encodings = torch.stack(table_encodings, dim=1)  # Shape: [batch_size, n_tables, hidden_dim]
        table_encodings = torch.concatenate([self.table_index_encoding.unsqueeze(0).expand(batch_size, -1, -1),
                                             table_encodings], dim=-1)
        
        # table_encodings = self.table_projects(table_encodings)


        # For each sample, select the corresponding template's query based on template_id
        template_ids = torch.tensor([self.id2index[id] for id in template_ids])
        selected_template_queries = self.template_queries[template_ids].unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
        

        merge_encodings = torch.concatenate([selected_template_queries, table_encodings], dim=1)
        merge_table_masks = torch.concatenate([torch.zeros(batch_size, 1, device=table_masks.device), ~table_masks], dim=-1)
        # Perform cross-attention with table_encoding as key and value, and template query as query
        attn_output = self.template_transformer(merge_encodings, src_key_padding_mask=merge_table_masks)[:, 0, :]  # Shape: [batch_size, hidden_dim]
        table_and_template_feature = torch.concatenate([attn_output, self.template_emb[template_ids]], dim=-1) # Shape: [batch_size, hidden_dim+table_index_encoding_len]

        logits = self.classification_head(table_and_template_feature)  # Apply the classification head to the attention output
                    
        return logits

    def get_other_attrs(self):
        return {
            'id2index': self.id2index,
            'template_ids': self.template_ids,
            'n_tables': self.n_tables,
            'n_templates': self.n_templates
        }

    def fix_parameters(self, other_attrs_dict, new_template_ids):
        for param in self.template_transformer.parameters():
            param.requires_grad = False
        
        for param in self.filter_projects.parameters():
            param.requires_grad = False

        self.template_emb.requires_grad = True
        
        # for k, v in other_attrs_dict.items():
        #     setattr(self, k, v)
        
        # self.old_template_queries = deepcopy(self.template_queries)
        
        # self.template_queries = nn.Parameter(torch.randn(len(new_template_ids), self.hidden_dim))
        # self.template_emb = nn.Parameter(torch.randn(len(new_template_ids), self.table_index_encoding_len))
        
        # self.id2index = {id: i for i, id in enumerate(new_template_ids)}





