from utils import *
from sklearn.metrics import mean_absolute_error


train, test, max_user, max_work = get_data()

#######################################################################
model = get_model_1(max_work, max_user)

history = model.fit([get_array(train["movieId"]), get_array(train["userId"])], get_array(train["rating"]), nb_epoch=10,
                    validation_split=0.2)

predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])

test_performance = mean_absolute_error(test["rating"], predictions)

print(" Test Mae model 1 : %s " % test_performance)

#######################################################################
model = get_model_2(max_work, max_user)

history = model.fit([get_array(train["movieId"]), get_array(train["userId"])], get_array(train["rating"]), nb_epoch=10,
                    validation_split=0.2)

predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])

test_performance = mean_absolute_error(test["rating"], predictions)

print(" Test Mae model 2 : %s " % test_performance)

#######################################################################
model = get_model_3(max_work, max_user)

history = model.fit([get_array(train["movieId"]), get_array(train["userId"])], get_array(train["rating"]), nb_epoch=10,
                    validation_split=0.2)

predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])

test_performance = mean_absolute_error(test["rating"], predictions)

print(" Test Mae model 3 : %s " % test_performance)