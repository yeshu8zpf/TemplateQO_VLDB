from preprocess_rank import split_train_data


rates = [2, 4, 6, 8, 10]
sql_path = 'data/imdb_10f/train/template0/sql.txt'
plan_path = 'data/imdb_10f/train/template0/plan.txt'
time_path = 'data/imdb_10f/train/template0/time.txt'

for rate in rates:
    split_train_data(sql_path=sql_path, time_path=time_path, plan_path=plan_path, rate=rate)
