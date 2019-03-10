# Recommender_keras

Description of the approach : https://medium.com/@CVxTz/movie-recommender-model-in-keras-e1d015a0f513

Requirements : Keras, tensorflow, pandas, numpy 


# Movie recommender model in Keras

![](https://cdn-images-1.medium.com/max/1600/1*hQAQ8s0-mHefYH83uDanGA.gif)
<span class="figcaption_hack">Source : Wikipedia</span>

### Objective

Recommender systems are models that are able to make recommendations to users
based on the history of the users’s behaviour. Collaborative filtering is one
way to build a recommender system that is based on the ratings of the users.The
objective is to build a simple collaborative filtering model using Keras.We will
be using the new MovieLens Dataset that has 100000 ratings, 9000 movie and 700
users Available here:
[https://grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens).

### Dataset

Users give a rating between 1 and 5 to some of the movies and the objective is
to predict the score they are going to give to other movies in the future in
order to recommend to them the ones they are most likely going to want to see.We
split the data by time so the all the samples in the train set were created
before every sample in the test set. This way we are effectively predicting the
future, similar to how the model might be used in production.

### Model 1

The first model we implement is a simple linear model where we learn a dense
representation for each movie and each user in our dataset.

![](https://cdn-images-1.medium.com/max/1600/1*Nx7DTTN3R3RnmTwWTGsfRg.png)

Code to do that is pretty straightforward with Keras :

This model does an element-wise multiplication of the movie vector and the user
vector then applies a linear layer in order to produce the predicted score.

The model achieves a mean_absolute_error of **0.98**. The mean_absolute_error is
the average absolute difference between the actual scores and the predicted
scores.

### Model 2

The first model does not explicitly take into account the bias that a user might
have in giving consistently high scores to every movie he watches or a movie
having consistently bad scores for all users. This is why we introduce a bias
for each of the users and for each movie.

![](https://cdn-images-1.medium.com/max/1600/1*-AKq5GrqrDMr5NDBs49jcA.png)

mean_absolute_error = **0.94**

This means that adding the bias helps improve the result ( Lower is better) a
tiny bit.

**Model 3**

We add a nonlinear fully connected layer in order to improve the expressivity
and complexity of our network.

![](https://cdn-images-1.medium.com/max/1600/1*abYDcq8_5Z_LxJ4TK1CMRg.png)

mean_absolute_error = **0.84**

This yield a significant improvement and we probably can further improve the
result if we properly tune the hyper-parameters of the network.

One other improvement we can make is to include contextual information like the
category of the movie or the description.

Code to reproduce the results is at :
[https://github.com/CVxTz/Recommender_keras](https://github.com/CVxTz/Recommender_keras)

