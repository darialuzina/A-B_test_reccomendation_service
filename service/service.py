### imports
 
import os
import hashlib
from typing import List
from datetime import datetime
from fastapi import FastAPI, Query 
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import create_engine
from schema import PostGet, Response

### constants for A/B test splitting
SALT = "some_random_salt"
SPLIT_RATIO = 0.5  # 50% test, 50% control


### function to determine user experiment group
def get_exp_group(user_id: int) -> str:
    hash_value = hashlib.md5(f"{user_id}{SALT}".encode()).hexdigest()
    return "test" if int(hash_value, 16) % 100 < (SPLIT_RATIO * 100) else "control"

### loading models

def get_model_path(path: str, name: str) -> str:
    if os.environ.get("IS_LMS") == "1":  
        MODEL_PATH = '/workdir/user_input/' + name
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models(model_name: str):
    if model_name == 'model_control':
        model_path = get_model_path("/Users/daraluzina/ML/HW_22/Ver_4.0/catboost_control.cbm", model_name)
    else:
        model_path = get_model_path("/Users/daraluzina/ML/HW_22/Ver_4.0/catboost_test.cbm", model_name)
    
    from_file = CatBoostClassifier()
    return from_file.load_model(model_path)

model_control = load_models("model_control")
model_test = load_models("model_test")


### loading features

import pandas as pd
from sqlalchemy import create_engine


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)



def load_features() -> pd.DataFrame:
    df_users = batch_load_sql("SELECT * from public.user_data")
    df_posts = batch_load_sql("SELECT * from daria_luzina_features_post_3")
    return df_users, df_posts

df_users, df_posts = load_features()



### function to transform datasets to predictions and recommendations

def posts_recommendation(user_id, model, n=5):
    df = df_users[df_users['user_id']==user_id]
    df = df.merge(df_posts.drop(columns='text'), how='cross')
    df = df.drop(columns = 'user_id')
    df = df.set_index('post_id')
    df['predict_proba'] = model.predict_proba(df)[:,1]
    df = df.sort_values(by='predict_proba', ascending=False).head(n)
    df = df.merge(df_posts, left_index=True, right_on = 'post_id').reset_index()
    df = df.rename(columns={"post_id": "id"})
    df = df.rename(columns={"topic_x": "topic"})
    df = df[['id','text','topic']]
    list_of_dicts = df.to_dict('records')
    return list_of_dicts


### endpoint
app = FastAPI()


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
    id: int,
    time: datetime = Query(...),
    limit: int = Query(5)
) -> Response:
    exp_group = get_exp_group(id) 
    model = model_test if exp_group == "test" else model_control  # choosing the correct model

    recommendations = posts_recommendation(id, model, limit)

    return Response(exp_group=exp_group, recommendations=recommendations)
