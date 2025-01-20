import argparse
import json
from time import sleep
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.PilotConfig import PostgreSQLConfig

def hint2plan(sql, hint, data_interactor: PilotDataInteractor):
    # Push the hint as a comment, and then retrieve execution time, physical plan, and records.
    data_interactor.push_pg_hint_comment(hint)
    data_interactor.pull_execution_time()
    data_interactor.pull_physical_plan()
    data_interactor.pull_record()
    
    result = data_interactor.execute(sql)
    plan = result.physical_plan
    time = result.execution_time
    plan['Execution Time'] = time
    value = result.records.values[0][0]
    return plan, time, value

def explainPlan(sql, hint, data_interactor, time):
    # Push the hint as a comment, then pull estimated cost and physical plan.
    data_interactor.push_pg_hint_comment(hint)
    data_interactor.pull_estimated_cost()
    data_interactor.pull_physical_plan()
    
    result = data_interactor.execute(sql)
    plan = result.physical_plan
    plan['Timeout'] = True
    plan['Execution Time'] = time
    return plan

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

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Command-line arguments for the script.")
    
    # Define command-line arguments
    parser.add_argument('--sql_path', type=str, required=True, help="Path to the SQL file")
    parser.add_argument('--hint_path', type=str, required=True, help="Path to the hint file")
    parser.add_argument('--plan_path', type=str, required=True, help="Directory to save the plan file")
    parser.add_argument('--time_path', type=str, required=True, help="Directory to save the time file")
    parser.add_argument('--database', type=str, required=True, default='imdb')
    parser.add_argument('--timeout', type=int, required=False, default=15*60)

    # Parse command-line arguments
    args = parser.parse_args()

    # Use the provided command-line arguments
    sql_path = args.sql_path
    hint_path = args.hint_path
    plan_path = args.plan_path
    time_path = args.time_path
    database = args.database
    timeout = args.timeout
    
    # Read SQL queries
    with open(sql_path, 'r') as f:
        sql_lines = f.readlines()
    
    qids = [line.strip().split('#####')[0] for line in sql_lines]
    sqls = [line.strip().split('#####')[1] for line in sql_lines]
    
    # Read hint list
    with open(hint_path) as f:
        hint_list = f.readlines()

    # Create the data interactor
    data_interactor = create_data_interactor(database, timeout=timeout)

    # Iterate over SQL queries
    for idx, sql in zip(qids, sqls):
        plan_list = []
        time_list = []

        # Iterate over hints
        for hint in hint_list:
            try:
                plan, time, value = hint2plan(sql, hint, data_interactor)
                sleep(0.1)
                plan_list.append(json.dumps([plan]))
                time_list.append(str(time))
                
            except Exception as e:
                print(e)
                # If error occurs, use the explainPlan method
                plan = explainPlan(sql, hint, data_interactor, timeout)
                plan_list.append(json.dumps([plan]))
                time_list.append(str(timeout))

        # Save the results to the plan file
        with open(plan_path, 'a') as f:
            f.write(f"{idx}#####{'#####'.join(plan_list)}\n")

        # Save the results to the time file
        with open(time_path, 'a') as f:
            f.write(f"{idx}#####{'#####'.join(time_list)}\n")