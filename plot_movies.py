from utils import *
import pickle
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


train, test, max_user, max_work, _ = get_data()

movies = pd.read_csv("data/movies.csv")

movie_title = dict(zip(movies["movieId"], movies["title"]))

model = get_model_1(max_user=max_user, max_work=max_work)
model.load_weights("model_1.h5")

embedding_work = model.get_layer("work").get_weights()[0]

print(embedding_work)

mapping_work = pickle.load(open("mapping_work.pkl", "rb"))

reverse_mapping = dict((v,k) for k,v in mapping_work.iteritems())

embedding = {}

for id in movie_title:
    if id in mapping_work:
        embedding[id] = embedding_work[mapping_work[id], :]


list_titles = []
list_embeddings = []

for id in embedding:
    list_titles.append(movie_title[id])
    list_embeddings.append(embedding[id])

matrix_embedding = np.array(list_embeddings)

X_embedded = TSNE(n_components=2).fit_transform(matrix_embedding)

vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]


data = [
    go.Scatter(
        x=vis_x,
        y=vis_y,
        mode='markers',
        text=list_titles
    )
]

layout = go.Layout(
    title='Movies'
)

fig = go.Figure(data=data, layout=layout)

plotly.offline.plot(fig, filename='movies.html')