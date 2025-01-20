import re
import numpy as np
import sqlglot
import json, pickle, random, os, sqlparse
from sqlparse.sql import Where, Identifier, IdentifierList, Token, Function, Parenthesis, Comparison

import torch
import torch.nn.functional as F

import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, Whitespace, Operator
import wandb

import re
import numpy as np
import sqlglot
import json, pickle, random, os, sqlparse
from sqlparse.sql import Where, Identifier, IdentifierList, Token, Function, Parenthesis, Comparison

import torch
import torch.nn.functional as F

import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, Whitespace, Operator
import wandb

def parse_conditions(token_list):
    """Recursively parse the WHERE clause and extract the condition list"""
    conditions = []
    # tokens = list(token_list.flatten())
    tokens = token_list.tokens
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]

        if token.ttype in [Whitespace, sqlparse.tokens.Newline] or token.ttype is Keyword and token.normalized == 'WHERE' \
            or token.ttype is Keyword and token.normalized == 'AND' \
            or isinstance(token, Identifier) \
            or token.normalized == ';':
            idx += 1
            continue


        # Handle 'IN' operator
        if token.ttype is Keyword and token.normalized == 'IN':
            # Find the column name to the left of 'IN'
            # Assume the column name is the closest identifier before the 'IN' operator
            left = ''
            for back_idx in range(idx - 1, -1, -1):
                back_token = tokens[back_idx]
                if isinstance(back_token, Identifier):
                    left = str(back_token).strip().lower()
                    break
                elif back_token.ttype not in [Whitespace, sqlparse.tokens.Punctuation]:
                    # Stop searching if a non-identifier and non-whitespace token is encountered
                    break

            # Find the Parenthesis after 'IN'
            in_values = []
            idx += 1
            while not isinstance(tokens[idx], Parenthesis) and idx < len(tokens):
                idx += 1
            parenthesis = tokens[idx]
            # Extract the values inside the parentheses
            values_str = parenthesis.value.strip("()")
            # Split the values using regex, considering commas in values
            in_values = [v.lstrip().lstrip("'").lower() for v in values_str.split("',")]
            in_values[-1] = in_values[-1].rstrip("'")
            idx += 1

            conditions.append((left, 'IN', in_values))
            continue

        # Check for comparison operators
        if isinstance(token, Comparison):
            left, op, right = token.normalized.split(' ')

            conditions.append((left, op, right.strip("'")))
            idx += 1
            continue

        # Handle other groupings (e.g., nested conditions)
        else:
            raise TypeError('token type')
        

    return conditions

def normalize_value(value, min_val, max_val):
    """Normalize a value to the range [0, 1]"""
    if max_val > min_val:
        return (value - min_val) / (max_val - min_val)
    else:
        return 0.0

def matches_pg_like_pattern(string, pattern):
    # Use rare Unicode characters as placeholders, assuming they are not in the pattern
    placeholder_for_percent = "\uFFFF1"
    placeholder_for_underscore = "\uFFFF2"
    
    # Replace '%' and '_' in the original pattern with placeholders
    pattern = pattern.replace('%', placeholder_for_percent)
    pattern = pattern.replace('_', placeholder_for_underscore)
    
    # Escape remaining characters in the pattern to treat special characters like parentheses, quotes as literals
    pattern = re.escape(pattern)
    
    # Replace the placeholders back with regex wildcards
    # Note: re.escape(placeholder) escapes the placeholders again, so we need to match the escaped form
    pattern = pattern.replace(re.escape(placeholder_for_percent), '.*')
    pattern = pattern.replace(re.escape(placeholder_for_underscore), '.')
    
    # Construct a full regex to match the entire string
    regex = f"^{pattern}$"
    return re.match(regex, string) is not None

def extract_values_from_in_filter(in_filter):
    """Extract the list of values from an 'IN' filter"""
    if isinstance(in_filter, list):
        return [v.lower() for v in in_filter]
    return []

def generate_binary_vector_with_nearest_edges(min_val, max_val, num_bins=100):
    """
    Generate a binary vector indicating the range [min_val, max_val] within [0, 1],
    where min_val and max_val are rounded to their nearest bin edges.

    Parameters:
        min_val (float): Minimum value in the range [0, 1].
        max_val (float): Maximum value in the range [0, 1].
        num_bins (int): Number of equally spaced bins (default is 100).
        
    Returns:
        numpy.ndarray: A binary vector of length num_bins.
    """
    # Ensure min_val and max_val are within [0, 1]
    assert 0 <= min_val <= 1, "min_val must be within [0, 1]"
    assert 0 <= max_val <= 1, "max_val must be within [0, 1]"
    assert min_val <= max_val, "min_val must be less than or equal to max_val"

    # Generate bin edges (quantiles)
    bins = np.linspace(0, 1, num_bins + 1)  # num_bins+1 because these are edges

    # Round min_val and max_val to the nearest bin edges
    min_index = np.abs(bins - min_val).argmin()
    max_index = np.abs(bins - max_val).argmin() - 1

    if min_index == max_index:
        max_index += 1

    # Create the binary vector
    binary_vector = np.zeros(num_bins, dtype=int)
    binary_vector[min_index:max_index] = 1

    return binary_vector

def get_max_num_columns(filter_columns):
    return max(len(v) for v in filter_columns.values())

def filter_to_table_column_encoding(filters, tables, filter_columns, encoding_leng=501, range_dict=None):
    max_num_columns = max(len(v) for v in filter_columns.values())
    table_column_encoding = np.zeros([len(tables), max_num_columns, encoding_leng])
    # The following block is commented out. It appears to be part of an earlier implementation for encoding ranges.
    # for tid, t in enumerate(tables):
    #     for cid, c in enumerate(filter_columns[t]):
    #         if isinstance(range_dict[t][c], dict):
    #             table_column_encoding[tid][cid][:-1] = 1
    #         else:
    #             table_column_encoding[tid][cid][1] = 1

    for table_name, col_name, info in filters:  
        table_id = tables.index(table_name)
        col_id = filter_columns[table_name].index(col_name)
        if isinstance(info, list):
            table_column_encoding[table_id][col_id][2] = 1
            table_column_encoding[table_id][col_id][0] = info[0]
            table_column_encoding[table_id][col_id][1] = info[1]
        elif isinstance(info, set):
            table_column_encoding[table_id][col_id][-1] = 1
            table_column_encoding[table_id][col_id][list(info)] = 1
    return table_column_encoding

def get_sql_tables_encoding(filter_infos, tables):
    sql_tables_encodings = []
    for filter_info in filter_infos:
        sql_tables = set()
        sql_tables.update(f[0] for f in filter_info)
        sql_tables_encoding = [0] * len(tables)
        for t in sql_tables:
            index = tables.index(t)
            sql_tables_encoding[index] = 1
        sql_tables_encodings.append(sql_tables_encoding)
    return sql_tables_encodings
    
def encode_sql_query(sql_query, range_dict):
    """
    Encode the conditions in a SQL query into vector representations, including join conditions and filter conditions,
    and return the encoding for the tables involved in the conditions.
    
    :param sql_query: SQL query string
    :param range_dict: Min and max values for each column, format: {'a': {'a1': [min, max], 'a2': [min, max]}, ...}
    :return: Encoded vector, including join condition encoding, filter condition encoding, and table involvement encoding
    """
    parsed = sqlparse.parse(sql_query)[0]
    from_seen = False
    actual_joins = []
    actual_filters = []
    alias2table = {}  # Local alias mapping
    involved_tables = set()  # Set to record all tables involved in conditions

    # Extract tables and aliases
    for token in parsed.tokens:
        if token.ttype is sqlparse.tokens.Whitespace:
            continue
        if token.ttype is sqlparse.tokens.DML and token.value.upper() == 'SELECT':
            continue
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
            from_seen = True
            continue
        if from_seen and (isinstance(token, sqlparse.sql.IdentifierList) or isinstance(token, sqlparse.sql.Identifier)):
            identifiers = [token] if isinstance(token, sqlparse.sql.Identifier) else token.get_identifiers()
            for idf in identifiers:
                table_name = idf.get_real_name().lower()
                alias = (idf.get_alias() or table_name).lower()
                alias2table[alias] = table_name
            from_seen = False  # Only process the FROM clause once
        if isinstance(token, sqlparse.sql.Where):
            # Parse WHERE conditions
            conditions = parse_conditions(token)
            for left, op, right in conditions:
                left = left.strip().lower()
                op = op.strip()
                right = right.strip().lower() if isinstance(right, str) else right
                # Check if it's a join condition
                if op == '=' and '.' in left and '.' in right:
                    left_table_alias = left.split('.')[0]
                    right_table_alias = right.split('.')[0]
                    left_col = alias2table.get(left_table_alias, left_table_alias) + '.' + left.split('.')[1]
                    right_col = alias2table.get(right_table_alias, right_table_alias) + '.' + right.split('.')[1]
                    join_condition = frozenset([left_col.lower(), right_col.lower()])
                    actual_joins.append(join_condition)
                    
                    # Add tables involved in joins
                    involved_tables.add(alias2table.get(left_table_alias, left_table_alias))
                    involved_tables.add(alias2table.get(right_table_alias, right_table_alias))
                else:
                    # Handle filter conditions
                    if '.' in left:
                        table_alias = left.split('.')[0]
                        actual_filters.append((left.lower(), op, right))
                        
                        # Add table involved in filters
                        involved_tables.add(alias2table.get(table_alias, table_alias))
            break  # Assume there is only one WHERE clause
    
    filter_columns = {}
    for ac, op, value in actual_filters:
        a, c = ac.split('.')
        t = alias2table[a] 
        if isinstance(range_dict[t][c], dict):
            if '.'.join([t,c]) not in filter_columns:
                filter_columns['.'.join([t,c])] = set()
            if op.lower() == 'like':
                regex_pattern = re.escape(value).replace('%', '.*')  # Convert to regex
                # Compile the regex pattern
                regex = re.compile(regex_pattern.lower())
                # Find the keys that do not match the pattern
                not_matching_key_ids = set([i for i, key in enumerate(list(range_dict[t][c].keys())) if not regex.search(key.lower())]) 
                assert len(not_matching_key_ids) < len(range_dict[t][c]) 
                filter_columns['.'.join([t,c])] = filter_columns['.'.join([t,c])] | not_matching_key_ids
            elif op.lower() == 'in':
                not_matching_key_ids = set([i for i, key in enumerate(list(range_dict[t][c].keys())) \
                                            if key.lower() not in list(map(lambda x: x.lower(), value))])
                assert len(range_dict[t][c]) - len(not_matching_key_ids) == len(value)
                filter_columns['.'.join([t,c])] = filter_columns['.'.join([t,c])] | not_matching_key_ids
        else:
            if '.'.join([t,c]) not in filter_columns:
                filter_columns['.'.join([t,c])] = [0, 1]
            norm_v = normalize_value(float(value), *range_dict[t][c])
            if op in ['>=', '>']:
                filter_columns['.'.join([t,c])][0] = norm_v
            elif op in ['<=', '<']:
                filter_columns['.'.join([t,c])][1] = norm_v
            elif op == '=':
                filter_columns['.'.join([t,c])][0] = norm_v
                filter_columns['.'.join([t,c])][1] = norm_v

    filter_infos = []
    for tc, info in filter_columns.items():
        t,c = tc.split('.')
        try:
            assert t  in range_dict
        except:
            assert 1

        if isinstance(range_dict[t][c], dict):
            info = set(range(len(range_dict[t][c]))) - info
        filter_infos.append([t, c, info])
    return filter_infos

def sql_to_table_column_encoding(sqls, range_dict, filter_columns, encoding_leng):
    tables = list(filter_columns.keys())
    filter_infos = [encode_sql_query(sql, range_dict) for sql in sqls]
    table_column_encodings = [filter_to_table_column_encoding(f, tables, filter_columns, encoding_leng, range_dict) for f in filter_infos]
    return table_column_encodings, filter_infos

def get_column_mask_old(range_dict):
    table_columns = {t: list(col_ranges.keys()) for t, col_ranges in range_dict.items()}
    max_num_columns = max([len(v) for k, v in table_columns.items()])
    column_mask = torch.zeros(len(table_columns), max_num_columns, dtype=torch.bool)
    for i, (t, cs) in enumerate(table_columns.items()):
        column_mask[i][:len(cs)] = 1
    return column_mask

def get_column_mask(filter_columns):
    max_num_columns = max([len(v) for v in filter_columns.values()])
    column_mask = torch.zeros(len(filter_columns), max_num_columns, dtype=torch.bool)
    for i, (t, cs) in enumerate(filter_columns.items()):
        column_mask[i][:len(cs)] = 1
    return column_mask

def extract_aliases(sql_query):
        """
        Encodes the conditions in an SQL query into a vector representation, including join conditions and filter conditions.
        Returns the tables involved in the conditions.

        :param sql_query: SQL query string
        :return: A tuple (table_names, aliases) containing the tables involved and their aliases
        """
        parsed = sqlparse.parse(sql_query)[0]
        from_seen = False

        # Track involved tables
        table_names = set()  # Used to record all tables involved in conditions
        aliases = set()  # Used to store aliases
        # 2. Extract tables and aliases
        for token in parsed.tokens:
            if token.ttype is sqlparse.tokens.Whitespace:
                continue
            if token.ttype is sqlparse.tokens.DML and token.value.upper() == 'SELECT':
                continue
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue
            if from_seen and (isinstance(token, sqlparse.sql.IdentifierList) or isinstance(token, sqlparse.sql.Identifier)):
                identifiers = [token] if isinstance(token, sqlparse.sql.Identifier) else token.get_identifiers()
                for idf in identifiers:
                    table_name = idf.get_real_name().lower()
                    alias = (idf.get_alias() or table_name).lower()
                    table_names.add(table_name)
                    aliases.add(alias)
                from_seen = False  # Process the FROM clause only once
            if isinstance(token, sqlparse.sql.Where):
                break

        return table_names, aliases

def get_labels(latency_dict, qids):
    def get_multi_label(times):
        min_time = min(times)
        label = [1. if time - min_time < 0.05 else 0. for time in times]
        return label
    labels = []
    for qid in qids:
        label = get_multi_label(latency_dict[qid])
        labels.append(label)
    return labels

def load_data(dir):
    sql_tables_path = os.path.join(dir, 'sql_tables.npy')
    table_col_encodings_path = os.path.join(dir, 'table_column_encodings.npz')
    labels_path = os.path.join(dir, 'labels.npy')
    masks_path = os.path.join(dir, 'masks.npy')
    weights_path = os.path.join(dir, 'weights.npy')
    qids_path = os.path.join(dir, 'qids')
    template_ids_path = os.path.join(dir, 'template_ids')
    with open(qids_path) as f:
        qids = json.load(f)
    with open(template_ids_path) as f:
        template_ids = json.load(f)
    with open(table_col_encodings_path, 'rb') as f:
        table_col_encodings = pickle.load(f)

    return np.load(sql_tables_path), table_col_encodings, np.load(labels_path), qids, template_ids, np.load(masks_path), np.load(weights_path)

def get_all_template_id(dir, used_templates):
    ids = []
    for dirpath, dirnames, filenames in os.walk(dir):
        if len(used_templates) == 0:
            used_templates = [dirname for dirname in dirnames if 'template' in dirname]
        for dirname in used_templates:
            id = int(re.findall(r'\d+', dirname)[0])
            ids.append(id)
        break
    return ids

def get_label_weights(Y, smooth_factor=0.2, weighting_type='top'):
    '''
    weighting_type: [top, class, latency, uniform]
    '''

    Y = np.array(Y)
    pairwise_label = np.sign(Y[:, :, np.newaxis] - Y[:, np.newaxis, :]) # shape (m, n, n)
    if weighting_type == 'class':
        m, n = Y.shape
    
        Y_expanded_i = Y[:, :, np.newaxis]  # shape (m, n, 1)
        Y_expanded_j = Y[:, np.newaxis, :]  # shape (m, 1, n)
        
        # Compare results
        labels = np.where(Y_expanded_i < Y_expanded_j, 1,
                        np.where(Y_expanded_i > Y_expanded_j, 2, 0))  # shape (m, n, n)
        
        # Initialize frequency table
        frequencies = np.zeros((n, n, 2), dtype=int)

        
        # Count frequency of labels 1 and 2
        for label in [1, 2]:
            mask = (labels == label)
            # Since mask shape is (m, n, n), we need to sum over the sample dimension
            frequencies[:, :, label - 1] = mask.sum(axis=0)
        sum_freq = frequencies.sum() 
        frequencies = frequencies / sum_freq

        # Step 2: Find the corresponding frequency based on labels
        frequencies_selected = np.where(labels == 1, frequencies[np.newaxis, :, :, 0],
                                        np.where(labels == 2, frequencies[np.newaxis, :, :, 1], 0))  # shape (m, n, n)

        # Step 3: Calculate weights
        # W[k, i, j] = 1 / (frequency + smooth_factor) if label > 0 else 0
        W = np.where(labels > 0, 1.0 / (frequencies_selected + smooth_factor), 0.0)  # shape (m, n, n)

        # Step 4: Normalize weights so that the sum of all weights equals m
        total_sum = W.sum()
        scaling_factor = m / total_sum
        W_scaled = W * scaling_factor
        weights = W_scaled

    elif weighting_type == 'uniform':
        weights = np.ones_like(pairwise_label) / (pairwise_label.shape[1]^2)
    elif weighting_type == 'top':
        m, n = Y.shape
        # Step 1: Compute ranks, starting from 1
        ranks = np.argsort(np.argsort(Y, axis=1), axis=1) + 1  # shape (m, n)
        ranks = ranks.astype(float)

        # Step 2: Compute the rank sums of all candidate pairs, shape (m, n, n)
        r_sum = ranks[:, :, np.newaxis] + ranks[:, np.newaxis, :]  # Broadcasting

        # Step 3: Calculate weight f = 1.0 / (r_sum / 10 + smooth_factor)
        W = 1.0 / (r_sum / 10.0 + smooth_factor)  # shape (m, n, n)

        # Step 4: Set diagonal elements to 0, as the weight for i == j should be 0
        # Create a diagonal mask with shape (n, n)
        diag_mask = np.eye(n, dtype=bool)
        # Set diagonal elements to 0 for each sample
        W[:, diag_mask] = 0

        # Step 5: Normalize weights so that the sum of weights for each row equals 1
        # Compute the weight sum for each row, shape (m, n, 1)
        total_sums = W.sum(axis=(1, 2), keepdims=True)  # shape (m, 1, 1)

        # Use np.divide for element-wise division, avoiding division by 0
        W_normalized = np.divide(W, total_sums, where=total_sums > 0)

        # For samples where the total sum of weights is 0, keep the weights as 0
        zero_mask = (total_sums[:, 0, 0] == 0)  # shape (m,)
        W_normalized[zero_mask, :, :] = 0
        weights = W_normalized
    elif weighting_type == 'latency':
        m, n = Y.shape
    
        # Step 1: Calculate the absolute value of score differences, shape (m, n, n)
        # Use broadcasting to compute |Y_i - Y_j|
        Y_expanded_i = Y[:, :, np.newaxis]  # shape (m, n, 1)
        Y_expanded_j = Y[:, np.newaxis, :]  # shape (m, 1, n)
        diff = np.abs(Y_expanded_i - Y_expanded_j)  # shape (m, n, n)
        
        # Step 2: Introduce a smoothing parameter to avoid weights being 0
        W = diff # shape (m, n, n)
        
        # Step 3: Set diagonal elements to 0 as we don't want to assign weights to itself
        # Create a diagonal mask with shape (n, n)
        diag_mask = np.eye(n, dtype=bool)  # shape (n, n)
        # Set diagonal elements to 0 for each sample
        W[:, diag_mask] = 0
        
        # Step 4: Normalize weights so that the sum of the entire n x n matrix for each sample equals 1
        # Compute the total weight sum for each sample, shape (m, 1, 1)
        total_sums = W.sum()  # shape (m, 1, 1)
        
        # Use np.divide for element-wise division, avoiding division by 0
        W_normalized = W / total_sums
        W_smoothed = np.where(W_normalized > 0, W_normalized + smooth_factor, W_normalized)
        W_smoothed = W_smoothed / W_smoothed.sum() * m

        weights = W_smoothed
    # else:
        
    #     freqs = (pairwise_label==1).sum(axis=0) / (pairwise_label.shape[0]) # n * n
    #     re_freqs = 1 - freqs
    #     weights = freqs / (freqs.sum() + re_freqs.sum())
    #     reweights = 1 / (freqs.sum() + re_freqs.sum()) - weights
    #     weights = weights + smooth_factor
    #     reweights = reweights + smooth_factor
    #     weights = weights / (weights.sum() + reweights.sum())
    #     reweights = reweights / (weights.sum() + reweights.sum())

    #     weights = np.where(pairwise_label > 0, reweights, weights)
    return weights


def prepare_data_multi_template(template_dir, dir, max_num_plans, range_dict, filter_columns, encoding_leng, used_templates=[], weighting_type='uniform',
                                smooth_factor=0.2):

    tables = list(filter_columns.keys())

    def prepare_template_data(sql_path, plan_path, template_id, time_path=None):
        with open(sql_path) as f:
            sql_lines = f.readlines()
        
        with open(plan_path) as f:
            plan_lines = f.readlines()
        
        if time_path:
            with open(time_path) as f:
                time_lines = f.readlines()
        sqls = [line.split('#####')[1] for line in sql_lines]
        # qids
        qids = [line.split('#####')[0] for line in sql_lines]
        qid2plan_lines = {line.split('#####')[0]: line for line in plan_lines}
        ordered_plan_lines = [qid2plan_lines[qid] for qid in qids]
        num_plans = [len(line.split('#####')[1:]) for line in plan_lines]

        # table_column_encodings
        table_column_encodings, filter_infos = sql_to_table_column_encoding(sqls, range_dict, filter_columns, encoding_leng)

        # sql_tables
        sql_tables = get_sql_tables_encoding(filter_infos, tables)

        # template_ids
        template_ids = [template_id] * len(qids)
        
        # labels and masks
        if time_path:
            latency_dict = {line.split('#####')[0]: list(map(float, line.strip().split('#####')[1:])) for line in time_lines} 
        else:
            latency_dict = get_latency_dict(ordered_plan_lines)
        labels = [latency_dict[qid] for qid in qids]


        weights  = get_label_weights(labels, weighting_type=weighting_type, smooth_factor=smooth_factor)

        padded_labels, masks = [], []
        for label, n in zip(labels, num_plans):
            padded_labels.append(np.concatenate([np.array(label), np.zeros(max_num_plans-n)]))
            mask = np.array([1]*n+[0]*(max_num_plans-n))
            masks.append(mask)

        return qids, template_ids, sql_tables, table_column_encodings, padded_labels, masks, weights

    qids, template_ids, sql_tables, table_column_encodings, labels, masks, weights = [], [], [], [], [], [], []
    for dirpath, dirnames, filenames in os.walk(template_dir):
        if len(used_templates) > 0:
            for dirname in dirnames:
                if 'template' in dirname and dirname in used_templates:
                    id = int(re.findall(r'\d+', dirname)[0])
                    sql_path = os.path.join(template_dir, dirname, 'sql.txt')
                    plan_path = os.path.join(template_dir, dirname, 'plan.txt')
                    time_path = os.path.join(template_dir, dirname, 'time.txt')
                    if not os.path.exists(time_path):
                        time_path = None
                    qid_list, template_id_list, sql_table_list, table_column_encoding_list, label_list, mask_list, weights_list \
                        = prepare_template_data(sql_path, plan_path, id, time_path)
                    qids.extend(qid_list)
                    template_ids.extend(template_id_list)
                    sql_tables.extend(sql_table_list)
                    table_column_encodings.extend(table_column_encoding_list)
                    labels.extend(label_list)
                    masks.extend(mask_list)
                    weights.extend(weights_list)
        else:
            for dirname in dirnames:
                if 'template' in dirname:
                    id = int(re.findall(r'\d+', dirname)[0])
                    sql_path = os.path.join(template_dir, dirname, 'sql.txt')
                    plan_path = os.path.join(template_dir, dirname, 'plan.txt')
                    time_path = os.path.join(template_dir, dirname, 'time.txt')
                    if not os.path.exists(time_path):
                        time_path = None
                    qid_list, template_id_list, sql_table_list, table_column_encoding_list, label_list, mask_list, weights_list \
                        = prepare_template_data(sql_path, plan_path, id, time_path)
                    qids.extend(qid_list)
                    template_ids.extend(template_id_list)
                    sql_tables.extend(sql_table_list)
                    table_column_encodings.extend(table_column_encoding_list)
                    labels.extend(label_list)
                    masks.extend(mask_list)
                    weights.extend(weights_list)

 
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, 'table_column_encodings.npz'), 'wb') as f:
        pickle.dump(table_column_encodings, f)

    # sql_tables
    np.save(os.path.join(dir, 'sql_tables.npy'), np.array(sql_tables))

    # qids
    with open(os.path.join(dir, 'qids'), 'w') as f:
        json.dump(qids, f, indent=4)

    # template_ids
    with open(os.path.join(dir, 'template_ids'), 'w') as f:
        json.dump(template_ids, f)
    
    # labels and masks
    weights = np.stack(weights, axis=0)
    np.save(os.path.join(dir, 'masks.npy'), np.array(masks))
    np.save(os.path.join(dir, 'labels.npy'), np.array(labels))
    np.save(os.path.join(dir, 'weights.npy'), weights)
    return np.array(sql_tables), table_column_encodings, np.array(labels), qids, template_ids, np.array(masks), weights

def get_latency_dict(lines=None, path=None, dir=None):
    if path is not None:
        with open(path) as f:
            lines = f.readlines()
    latency_dict = {line.split('#####')[0]: [json.loads(plan)[0]['Execution Time'] for plan in line.split('#####')[1:]] for line in lines}
    first_length = len(next(iter(latency_dict.values())))

    # Assert that all lists have the same length
    assert all(len(value) == first_length for value in latency_dict.values())

    return latency_dict

def get_template_id(qid2template_id_dict, sql_path):
    with open(sql_path) as f:
        sql_lines = f.readlines()
    sql_qids = [line.split('#####')[0] for line in sql_lines]
    template_ids = [qid2template_id_dict[qid] for qid in sql_qids]
    return template_ids

def get_qid2template_id_dict(sql_path_list):
    qid2template_id_dict = {}
    for i, sql_path in enumerate(sql_path_list):
        with open(sql_path) as f:
            sql_lines = f.readlines()
        sql_qids = [line.split('#####')[0] for line in sql_lines]
        qid2template_id_dict.update({qid:i for qid in sql_qids})
    return qid2template_id_dict




def split_sqls(sql_path, train_sql_path, test_sql_path, valid_sql_path, plan_path, train_plan_path, test_plan_path, valid_plan_path,
               time_path, train_time_path, test_time_path, valid_time_path, train_rate, test_rate):
    # Read SQL lines from the provided path
    with open(sql_path) as f:
        lines = f.readlines()
    
    # Shuffle the lines randomly
    random.shuffle(lines)
    
    # Create directories for the output plan and time files if they do not exist
    os.makedirs(os.path.dirname(train_plan_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_plan_path), exist_ok=True)
    
    if valid_plan_path:
        os.makedirs(os.path.dirname(valid_plan_path), exist_ok=True)
    
    # Write train, test, and validation SQL lines based on the specified rates
    with open(train_sql_path, 'w') as f:
        f.writelines(lines[:int(len(lines) *train_rate)])
    with open(test_sql_path, 'w') as f:
        f.writelines(lines[int(len(lines) * train_rate):int(len(lines) * (train_rate+test_rate))])
    if valid_plan_path:
        with open(valid_sql_path, 'w') as f:
            f.writelines(lines[int(len(lines) * (train_rate+test_rate)):])

    # Extract the query IDs from the train, test, and validation SQL lines
    train_qids = [line.split('#####')[0] for line in lines[:int(len(lines) *train_rate)]]
    test_qids = [line.split('#####')[0] for line in lines[int(len(lines) * train_rate):int(len(lines) * (test_rate+train_rate))]]
    if valid_plan_path:
        valid_qids = [line.split('#####')[0] for line in lines[int(len(lines) * (test_rate+train_rate)):]]


    # Read the plan lines
    with open(plan_path, 'r') as f:
        plan_lines = f.readlines()

    # Create a dictionary mapping query IDs to plan lines
    qid2plan_lines = {line.split('#####')[0]: line for line in plan_lines}
    
    # Get the corresponding plan lines for train, test, and validation sets
    train_plan_lines = [qid2plan_lines[qid] for qid in train_qids]
    test_plan_lines = [qid2plan_lines[qid] for qid in test_qids]
    if valid_plan_path:
        valid_plan_lines = [qid2plan_lines[qid] for qid in valid_qids]

    # Ensure the directories for train plan are created
    os.makedirs(os.path.dirname(train_plan_path), exist_ok=True)

    # Write the plan lines to the respective files
    with open(train_plan_path, 'w') as f:
        f.writelines(train_plan_lines)
    with open(test_plan_path, 'w') as f:
        f.writelines(test_plan_lines)
    if valid_plan_path:
        with open(valid_plan_path, 'w') as f:
            f.writelines(valid_plan_lines)

    # Read the time lines
    with open(time_path, 'r') as f:
        time_lines = f.readlines()
    
    # Create a dictionary mapping query IDs to time lines
    qid2time_lines = {line.split('#####')[0]: line for line in time_lines}
    
    # Get the corresponding time lines for train, test, and validation sets
    train_time_lines = [qid2time_lines[qid] for qid in train_qids]
    test_time_lines = [qid2time_lines[qid] for qid in test_qids]
    if valid_plan_path:
        valid_time_lines = [qid2time_lines[qid] for qid in valid_qids]
    
    # Write the time lines to the respective files
    with open(train_time_path, 'w') as f:
        f.writelines(train_time_lines)
    with open(test_time_path, 'w') as f:
        f.writelines(test_time_lines)
    if valid_plan_path:
        with open(valid_time_path, 'w') as f:
            f.writelines(valid_time_lines)

def print_best_latency_sum(path):
    # Get the latency dictionary from the specified path
    latency_dict = get_latency_dict(path=path)
    sum = 0
    # Sum the minimum latency for each query
    for v in latency_dict.values():
        sum += min(v)
    print(sum)

def get_filter_col_types(range_dict, filter_columns):
    # Get the column types based on the filter conditions
    types_list = []
    for t, cs in filter_columns.items():
        types = []
        for c in cs:
            if isinstance(range_dict[t][c], dict):
                types.append(1)  # Type 1 for range-based filter
            else:
                types.append(0)  # Type 0 for non-range-based filter
        types_list.append(types)
    return types_list

def merge_for_lero(dir, index_list, output_plan_path, output_sql_path):
    # Merge plan files for the specified templates
    output_plan_lines = []
    for i in index_list:
        with open(os.path.join(dir, f'template{i}/plan.txt')) as f:
            output_plan_lines.extend(f.readlines())
    
    # Write the merged plan lines to the output file
    with open(output_plan_path, 'w') as f:
        f.writelines(output_plan_lines)

    # Merge SQL files for the specified templates
    output_sql_lines = []
    for i in index_list:
        with open(os.path.join(dir, f'template{i}/sql.txt')) as f:
            output_sql_lines.extend(f.readlines())
    
    # Write the merged SQL lines to the output file
    with open(output_sql_path, 'w') as f:
        f.writelines(output_sql_lines)

def plan2alias_seq(tables, plan):
    """
    Generate fixed-length partition labels and join type labels based on PostgreSQL execution plans,
    and generate a string representation of the join sequence.

    Parameters:
    - tables (list of str): List of table names, with fixed order.
    - plan (dict): PostgreSQL execution plan, typically from EXPLAIN (FORMAT JSON).

    Returns:
    - partition_labels (list of list of int): Fixed-length label vectors (0, 1, -1) for each partition step.
    - join_type_labels (list of int): Join type labels for each partition step.
    - join_sequence (str): String representation of the join sequence.
    """
    # partition_labels = []
    # join_type_labels = []

    # Define join type mapping
    join_type_map = {
        'Nested Loop': 0,
        'Hash Join': 1,
        'Merge Join': 2
    }

    join_str_map = {
        0: 'Nested_Loop',
        1: 'Hash_Join',
        2: 'Merge_Join'
    }

    # Define supported scan node types
    scan_node_types = [
        'Seq Scan', 'Index Scan', 'Bitmap Heap Scan', 
        'Bitmap Index Scan', 'Index Only Scan'
    ]

    def traverse(node):
        """
        Recursively traverse the execution plan tree to collect partition steps and join types.

        Parameters:
        - node (dict): The current node's execution plan.

        Returns:
        - tables_in_subtree (set of int): The set of table indices involved in the current subtree.
        - join_sequence_part (str): The join sequence part for the current node.
        """
        node_type = node.get('Node Type')

        # Handle join nodes
        if node_type in join_type_map.keys():
            # Get join type
            join_type = join_type_map.get(node_type, -1)
            if join_type == -1:
                raise ValueError(f"Unknown join type: {node_type}")

            # Get left and right child nodes
            plans = node.get('Plans', [])
            if len(plans) != 2:
                raise ValueError(f"{node_type} node requires two child nodes, but {len(plans)} were found.")

            left_node = plans[0]
            right_node = plans[1]

            # Recursively get tables and join sequences from left and right subtrees
            left_tables, left_join_sequence = traverse(left_node)
            right_tables, right_join_sequence = traverse(right_node)

            # Create label list, initialize with -1
            labels = [-1] * len(tables)

            # Assign label 0 for left subtree tables, label 1 for right subtree tables
            for table_idx in left_tables:
                labels[table_idx] = 0
            for table_idx in right_tables:
                labels[table_idx] = 1

            # Add current step's labels and join type
            # partition_labels.append(labels)
            # join_type_labels.append(join_type)

            # Build the current join sequence representation
            left_str = f"{left_join_sequence}" if left_join_sequence else f"{' '.join([tables[idx] for idx in left_tables])}"
            right_str = f"{right_join_sequence}" if right_join_sequence else f"{' '.join([tables[idx] for idx in right_tables])}"
            join_str = f"( {left_str} {join_str_map[join_type]} {right_str} )"

            # Return the tables involved in the current subtree and the join sequence part
            return left_tables.union(right_tables), join_str

        # Handle scan nodes (leaf nodes)
        elif node_type in scan_node_types:
            # Get table name
            relation_name = node.get('Relation Name')
            alias = node.get('Alias')
            if not alias:
                raise KeyError
            table_name = relation_name

            if table_name is None:
                raise ValueError("Scan node missing 'Relation Name' or 'Alias'.")

            try:
                table_idx = tables.index(table_name)
            except ValueError:
                raise ValueError(f"Table '{table_name}' not found in table list: {tables}")

            # Return the index set of the current table and the alias for the node
            return {table_idx}, alias

        # Handle other node types (assumes only one child node)
        elif node_type in ["Aggregate", "Append", "Gather", "Gather Merge", "Group", "Hash", "Limit", "LockRows", 
                           "Materialize", "ProjectSet", "Recursive Union", "Sort", "Subquery Scan", "Unique", "WindowAgg"]:
            plans = node.get('Plans', [])
            if len(plans) == 0:
                raise ValueError(f"{node_type} node has no child nodes.")
            elif len(plans) == 1:
                return traverse(plans[0])
            else:
                raise ValueError(f"{node_type} node has multiple child nodes, not defined how to handle.")

        else:
            # For unknown node types, we can choose to ignore or log them
            # Here we choose to ignore and return an empty set
            print(f"Warning: Encountered unknown node type: {node_type}, ignoring this node.")
            return set(), ""


    # Start traversing the execution plan tree
    _, join_sequence = traverse(plan)

    return join_sequence

def save_checkpoint(model, optimizer, config, logger):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'other_attr_dict': model.get_other_attrs(),
        'opt_state_dict': optimizer.state_dict()
        }
    os.makedirs(os.path.dirname(config.log.model_save_path), exist_ok=True)
    torch.save(checkpoint, config.log.model_save_path)
    logger.info(f"Model saved at {config.log.model_save_path}")

def log_wandb(first_template_latencys, latency_sum_list, valid_latency_list, config, first_template_id, test_acc_list, table, epoch):
    first_template_latency = sum(first_template_latencys[-10:]) / 10
    min_latency = min(latency_sum_list)
    mean_latency = sum(latency_sum_list[-250:]) / 250 
    test_latency_with_best_valid = latency_sum_list[valid_latency_list.index(min(valid_latency_list))]

    window = np.ones(10)
    # Use convolution to calculate the sum of all windows
    valid_latency_list = torch.tensor(valid_latency_list, dtype=torch.float32)
    window = torch.tensor(window, dtype=torch.float32).view(1, 1, -1)  # Adjusting for conv1d input format

    # Perform convolution
    window_sums = F.conv1d(valid_latency_list.view(1, 1, -1), window, padding=0).squeeze()

    # Find the starting index of the window with the minimum sum
    min_sum_index = torch.argmin(window_sums)
    
    # Generate a list of indices for the minimum sum window
    min_indices = list(range(min_sum_index, min_sum_index + 10))
    test_latency_with_best_mean_valid = sum([latency_sum_list[i] for i in min_indices]) / 10
    if config.log.wandb:
        wandb.log({
            f'test/template{first_template_id}_mean_latency': first_template_latency,
            'min_latency': min_latency,
            'mean_latency': mean_latency,
            'test_latency_with_best_valid': test_latency_with_best_valid,
            'test_latency_with_best_mean_valid': test_latency_with_best_mean_valid,
            'latency_table': table,
            'best_best_acc': max(test_acc_list)
        }, step=epoch)


def split_train_data(plan_path, time_path, sql_path, rate):
    def check(plan_path, time_path, sql_path):
        with open(plan_path) as f:
            plan_lines = f.readlines()
            plan_qids = [line.split('#####')[0] for line in plan_lines]

        with open(time_path) as f:
            time_lines = f.readlines()
            time_qids = [line.split('#####')[0] for line in time_lines]
        
        with open(sql_path) as f:
            sql_lines = f.readlines()
            sql_qids = [line.split('#####')[0] for line in sql_lines]
        
        # Ensure that the lengths of all files are the same
        assert len(plan_qids) == len(time_qids) and len(plan_qids) == len(sql_qids)

        # Ensure that the query IDs match across the three files
        for qid1, qid2, qid3 in zip(plan_qids, time_qids, sql_qids):
            assert qid1 == qid2 and qid1 == qid3

    check(plan_path, time_path, sql_path)
    
    # Split SQL data based on the rate provided and save to a new location
    with open(sql_path) as f:
        sql_lines = f.readlines()
    remain_lines = sql_lines[:int(rate * len(sql_lines))]
    new_sql_path = os.path.join(os.path.dirname(sql_path), '..', f'template{rate}', 'sql.txt')
    os.makedirs(os.path.join(os.path.dirname(sql_path), '..', f'template{rate}'))
    with open(new_sql_path, 'w') as f:
        f.writelines(remain_lines)

    # Split time data based on the rate provided and save to a new location
    with open(time_path) as f:
        time_lines = f.readlines()
    remain_lines = time_lines[:int(rate * len(time_lines))]
    new_time_path = os.path.join(os.path.dirname(sql_path), '..', f'template{rate}', 'time.txt')
    with open(new_time_path, 'w') as f:
        f.writelines(remain_lines)
    
    # Split plan data based on the rate provided and save to a new location
    with open(plan_path) as f:
        plan_lines = f.readlines()
    remain_lines = plan_lines[:int(rate * len(plan_lines))]
    new_plan_path = os.path.join(os.path.dirname(sql_path), '..', f'template{rate}', 'plan.txt')
    with open(new_plan_path, 'w') as f:
        f.writelines(remain_lines)

def prepare_weights(template_dir, used_templates, weighting_type, smooth_factor):
    weights = []
    
    # Walk through the template directory and calculate weights for the selected templates
    for dirpath, dirnames, filenames in os.walk(template_dir):
        if len(used_templates) > 0:
            for dirname in dirnames:
                if 'template' in dirname and dirname in used_templates:
                    sql_path = os.path.join(template_dir, dirname, 'sql.txt')
                    time_path = os.path.join(template_dir, dirname, 'time.txt')
                    with open(sql_path) as f:
                        sql_lines = f.readlines()
                    qids = [line.split('#####')[0] for line in sql_lines]
                    with open(time_path) as f:
                        time_lines = f.readlines()
                    # Create a dictionary mapping query ID to its respective latency values
                    latency_dict = {line.split('#####')[0]: list(map(float, line.strip().split('#####')[1:])) for line in time_lines}
                    labels = [latency_dict[qid] for qid in qids]
                    # Calculate the template weights
                    template_weights = get_label_weights(labels, weighting_type=weighting_type, smooth_factor=smooth_factor)
                    weights.extend(template_weights)
        else:
            for dirname in dirnames:
                if 'template' in dirname:
                    sql_path = os.path.join(template_dir, dirname, 'sql.txt')
                    time_path = os.path.join(template_dir, dirname, 'time.txt')
                    with open(sql_path) as f:
                        sql_lines = f.readlines()
                    qids = [line.split('#####')[0] for line in sql_lines]
                    with open(time_path) as f:
                        time_lines = f.readlines()
                    # Create a dictionary mapping query ID to its respective latency values
                    latency_dict = {line.split('#####')[0]: list(map(float, line.strip().split('#####')[1:])) for line in time_lines}
                    labels = [latency_dict[qid] for qid in qids]
                    # Calculate the template weights
                    template_weights = get_label_weights(labels, weighting_type=weighting_type, smooth_factor=smooth_factor)
                    weights.extend(template_weights)
    
    # Stack the weights and return
    weights = np.stack(weights)
    return weights

def extract_cost_from_plan_txt(path, output_path):
    # Extract the cost information from the execution plan file
    with open(path) as f:
        lines = f.readlines()
    costs_lines = []
    for line in lines:
        arr = line.split('#####')
        qid = arr[0]
        plans = arr[1:]
        # Extract the 'Total Cost' from the JSON-formatted plan
        costs = [str(json.loads(p.strip())[0]['Plan']['Total Cost']) for p in plans]
        costs_lines.append('#####'.join([qid, *costs]) + '\n')
    
    # Save the costs to a new file
    with open(output_path, 'w') as f:
        f.writelines(costs_lines)
    
    return costs_lines

if __name__ == '__main__':
    path = "data/imdb_10d/train/template0/plan.txt"
    output_path = 'data/imdb_10d_pretrain/train/time.txt'
    cost_lines = extract_cost_from_plan_txt(path, output_path)

    

    # for i in [0,5,6,7,8,9]:
    #     dir = '/home/zpf/join_order/data/multi_template/stats'
    #     sql_path = f'{dir}/t{i}/sql.txt'
    #     plan_path = f"{dir}/t{i}/plan.txt"
    #     train_sql_path = f'data/stats/train/template{i}/sql.txt'
    #     train_plan_path = f'data/stats/train/template{i}/plan.txt'
    #     test_sql_path = f'data/stats/test/template{i}/sql.txt'
    #     test_plan_path = f'data/stats/test/template{i}/plan.txt'
    #     valid_sql_path = f'data/stats/valid/template{i}/sql.txt'
    #     valid_plan_path = f'data/stats/valid/template{i}/plan.txt'
    #     time_path = f'{dir}/t{i}/time.txt'
    #     train_time_path = f'data/stats/train/template{i}/time.txt'
    #     test_time_path = f'data/stats/test/template{i}/time.txt'
    #     valid_time_path = f'data/stats/valid/template{i}/time.txt'
    #     split_sqls(sql_path, train_sql_path, test_sql_path, None, 
    #                plan_path, train_plan_path, test_plan_path, None,
    #                time_path, train_time_path, test_time_path, None,
    #                train_rate=0.8, test_rate=0.2)
    # merge_for_lero('data/stats/train', [0,1,2,3,4,5,6,7,9,10], 'data/stats/train/train.txt', 'data/stats/train/sql.txt')
    # merge_for_lero('data/stats/test', [0,1,2,3,4,5,6,7,9,10], 'data/stats/test/test.txt', 'data/stats/test/sql.txt')
    # split_sqls('/home/zpf/join_order/data/multi_template/stats-10/t1_d/sql.txt', 'data/stats_10f/train/template0/sql.txt', 'data/stats_10f/test/template0/sql.txt', None,
    #            '/home/zpf/join_order/data/multi_template/stats-10/t1_d/plan.txt', 'data/stats_10f/train/template0/plan.txt', 'data/stats_10f/test/template0/plan.txt', None,
    #            '/home/zpf/join_order/data/multi_template/stats-10/t1_d/time.txt', 'data/stats_10f/train/template0/time.txt', 'data/stats_10f/test/template0/time.txt', None,
    #            train_rate=0.8, test_rate=0.2
    #            )
    # print_best_latency_sum('data/stats/test/test.txt')
    1