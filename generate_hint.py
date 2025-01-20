import argparse
from collections import defaultdict
import json
import random
import re
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.PilotConfig import PostgreSQLConfig

def generate_pg_hint(hint_string):
    # Define the mapping of join types
    join_mapping = {
        "Nested_Loop": "NestLoop",
        "Hash_Join": "HashJoin",
        "Merge_Join": "MergeJoin",
        # Add more join types as needed
    }

    # Define a simple tree node structure
    class JoinNode:
        def __init__(self, left, join_type, right):
            self.left = left
            self.join_type = join_type
            self.right = right

    # Parse the string into a tree structure
    def parse(tokens):
        def helper(it):
            try:
                token = next(it)
            except StopIteration:
                return None

            if token == '(':
                left = helper(it)
                join_type = next(it)
                right = helper(it)
                closing = next(it)
                if closing != ')':
                    raise ValueError("Unmatched parentheses")
                return JoinNode(left, join_type, right)
            elif token == ')':
                return None
            else:
                return token

        return helper(iter(tokens))

    # Generate Leading hints recursively
    def generate_leading(node):
        if isinstance(node, JoinNode):
            left = generate_leading(node.left)
            right = generate_leading(node.right)
            return f"({left} {right})"
        else:
            return node

    # Traverse the tree and collect join hints
    def traverse(node, hints):
        if isinstance(node, JoinNode):
            traverse(node.left, hints)
            traverse(node.right, hints)
            join_type = join_mapping.get(node.join_type, node.join_type)
            
            if isinstance(node.left, JoinNode) or isinstance(node.right, JoinNode):
                def collect_tables(n):
                    if isinstance(n, JoinNode):
                        return collect_tables(n.left) + collect_tables(n.right)
                    else:
                        return [n]

                left_tables = collect_tables(node.left)
                right_tables = collect_tables(node.right)
                all_tables = left_tables + right_tables
                seen = set()
                all_tables_unique = [x for x in all_tables if not (x in seen or seen.add(x))]
                hints.append(f"{join_type}({ ' '.join(all_tables_unique) })")
            else:
                hints.append(f"{join_type}({node.left} {node.right})")

    # Preprocess the string, ensuring parentheses are separated by spaces
    hint_string = hint_string.replace('(', ' ( ').replace(')', ' ) ')

    # Tokenize, keeping parentheses as separate tokens
    tokens = re.findall(r'\(|\)|\S+', hint_string)

    try:
        tree = parse(iter(tokens))
    except Exception as e:
        raise ValueError(f"Failed to parse hint string: {e}")

    if tree is None:
        raise ValueError("Empty or invalid hint string.")

    # Collect all join hints
    join_hints = []
    traverse(tree, join_hints)

    # Generate Leading hints
    leading = generate_leading(tree)
    leading = f"({leading})"

    # Construct the final pg_hint string
    pg_hint = "/*+ " + " ".join(join_hints) + f" Leading{leading}" + " */"
    return pg_hint

def plan2seq(tables, plan):
    """
    Generate fixed-length partition labels, join type labels, and a join sequence string from a PostgreSQL execution plan.

    Arguments:
    - tables (list of str): List of table names, order is fixed.
    - plan (dict): PostgreSQL execution plan, usually from EXPLAIN (FORMAT JSON).

    Returns:
    - partition_labels (list of list of int): Fixed-length label vectors (0, 1, -1) for each partition step.
    - join_type_labels (list of int): Labels for the join type at each partition step.
    - join_sequence (str): String representation of the join sequence.
    """
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

    scan_node_types = [
        'Seq Scan', 'Index Scan', 'Bitmap Heap Scan', 
        'Bitmap Index Scan', 'Index Only Scan'
    ]

    def traverse(node):
        """
        Recursively traverse the execution plan tree to collect partition steps and join types.

        Arguments:
        - node (dict): The current node in the execution plan.

        Returns:
        - tables_in_subtree (set of int): Set of table indices involved in the current subtree.
        - join_sequence_part (str): The join sequence part for the current node.
        """
        node_type = node.get('Node Type')

        # Handle join nodes
        if node_type in join_type_map.keys():
            join_type = join_type_map.get(node_type, -1)
            if join_type == -1:
                raise ValueError(f"Unknown join type: {node_type}")

            plans = node.get('Plans', [])
            if len(plans) != 2:
                raise ValueError(f"{node_type} node requires two child nodes, but got {len(plans)}.")

            left_node = plans[0]
            right_node = plans[1]

            left_tables, left_join_sequence = traverse(left_node)
            right_tables, right_join_sequence = traverse(right_node)

            labels = [-1] * len(tables)

            for table_idx in left_tables:
                labels[table_idx] = 0
            for table_idx in right_tables:
                labels[table_idx] = 1

            left_str = f"{left_join_sequence}" if left_join_sequence else f"{' '.join([tables[idx] for idx in left_tables])}"
            right_str = f"{right_join_sequence}" if right_join_sequence else f"{' '.join([tables[idx] for idx in right_tables])}"
            join_str = f"( {left_str} {join_str_map[join_type]} {right_str} )"

            return left_tables.union(right_tables), join_str

        # Handle scan nodes (leaf nodes)
        elif node_type in scan_node_types:
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

            return {table_idx}, alias

        # Handle other node types (assumed to have one child node)
        elif node_type in ["Aggregate", "Append", "Gather", "Gather Merge", "Group", "Hash", "Limit", "LockRows", 
                           "Materialize", "ProjectSet", "Recursive Union", "Sort", "Subquery Scan", "Unique", "WindowAgg"]:
            plans = node.get('Plans', [])
            if len(plans) == 0:
                raise ValueError(f"{node_type} node has no child nodes.")
            elif len(plans) == 1:
                return traverse(plans[0])
            else:
                raise ValueError(f"{node_type} node has multiple child nodes, handling not defined.")

        else:
            print(f"Warning: Unknown node type encountered: {node_type}, skipping node.")
            return set(), ""

    _, join_sequence = traverse(plan)

    return join_sequence

def create_data_interactor(database, config_file='configs/config.json', timeout: int=30*60):
    # Read the configuration from the provided config file
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    db_port = config_data.get('db_port', '5432')  # Default to 5432 if not found
    db_user = config_data.get('db_user', 'pilotscope')  # Default to 'pilotscope'
    db_user_pwd = config_data.get('db_user_pwd', 'pilotscope')  # Default to 'pilotscope'
    pg_bin_path = config_data.get('pg_bin_path', '$PG_PATH/bin')  # Default to '$PG_PATH/bin'
    pg_data_path = config_data.get('pg_data_path', '$PG_DATA')  # Default to '$PG_DATA'
    
    if timeout < 1 or timeout > 30*60:
        timeout = 30*60
    
    config = PostgreSQLConfig(
        db_port=db_port, 
        db_user=db_user, 
        db_user_pwd=db_user_pwd, 
        db=database, 
        sql_execution_timeout=timeout
    )
    
    config.enable_deep_control_local(pg_bin_path, pg_data_path)
    data_interactor = PilotDataInteractor(config)
    return data_interactor

import argparse
from collections import defaultdict
import json

# Function to get true cardinalities for the query
def get_true_cards(q, data_interactor: PilotDataInteractor):
    true_cards = {}
    try:
        data_interactor.pull_subquery_card(enable_parameterized_subquery=False)
        result = data_interactor.execute(q)
    except Exception as e:
        print(e)
        return None
    
    for k in result.subquery_2_card.keys():
        data_interactor.pull_record()
        try:
            res = data_interactor.execute(k)
            card = int(res.records.values[0][0])
            true_cards[k] = card
        except Exception as e:
            print(e)
            return None
    
    return true_cards

# Function to get the best plan based on true cardinalities
def get_best_plan_by_true_cards(q, true_cards, data_interactor: PilotDataInteractor):
    data_interactor.push_card(true_cards)
    data_interactor.pull_execution_time()
    data_interactor.pull_physical_plan()
    try:
        best_result = data_interactor.execute(q)
        return best_result.physical_plan['Plan'], best_result.execution_time
    except Exception as e:
        print(e)
        return None, None

# Main function to parse arguments, read files, and generate hints
if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Command-line arguments for the script.")

    # Define command-line arguments
    parser.add_argument('--sql_path', type=str, required=True, help="Path to the SQL file")
    parser.add_argument('--hint_path', type=str, required=True, help="Path to the hint file")
    parser.add_argument('--alias_map_path', type=str, required=True, help="Path to the alias map file")
    parser.add_argument('--database', type=str, required=True, help="Database name", default='imdb')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Assign parsed arguments to the respective variables
    sql_path = args.sql_path
    hint_path = args.hint_path
    alias_map_path = args.alias_map_path
    database = args.database

    # Output the values (for testing purposes)
    print(f"SQL Path: {sql_path}")
    print(f"Hint Path: {hint_path}")
    print(f"Alias Map Path: {alias_map_path}")
    print(f"Database: {database}")

    # Load alias map from the given path
    with open(alias_map_path) as f:
        alias_map = json.load(f)
    tables = list(alias_map.keys())

    # Read SQL queries
    with open(sql_path, 'r') as f:
        sql_lines = f.readlines()
    
    # Extract query IDs and SQL statements
    qids = [line.strip().split('#####')[0] for line in sql_lines]
    sqls = [line.strip().split('#####')[1] for line in sql_lines]
    
    # Randomly select 10% of the indices
    num = min(len(qids), 10)
    indices = random.sample(range(len(qids)), num)

    # Get the corresponding query IDs and SQL statements based on the selected indices
    sampled_qids = [qids[i] for i in indices]
    sampled_sqls = [sqls[i] for i in indices]
    
    # Create the data interactor
    data_interactor = create_data_interactor(database)
    hint_dict = defaultdict(int)

    # Loop through each query, compute the true cardinalities, get the best plan, and generate hints
    for qid, sql in zip(sampled_qids, sampled_sqls):
        true_cards = get_true_cards(sql, data_interactor)
        if true_cards is None:
            continue 
        
        best_plan, _ = get_best_plan_by_true_cards(sql, true_cards, data_interactor)
        if best_plan is None:
            continue 

        seq = plan2seq(tables, best_plan)
        hint = generate_pg_hint(seq)

        # Track the frequency of each generated hint
        if hint in hint_dict:
            hint_dict[hint] += 1
        else:
            hint_dict[hint] = 1
    
    # Sort the hints by frequency and select the top 10
    top_10 = sorted(hint_dict.items(), key=lambda x: x[1], reverse=True)[:num]
    candidate_hints = [item[0].strip() + '\n' for item in top_10]

    # Save the top 10 hints to the specified file
    with open(hint_path, 'w') as f:
        f.writelines(candidate_hints)
