# Movie-Rating-Generator
Develop a Collaborative Filtering system to predict as accurately as possible the user item ratings. 


Collaborative Filtering (CF) systems measure similarity of users by their item preferences
and/or measure similarity of items by the users who like them. For this CF systems extract
Item profiles and user profiles and then compute similarity of rows and columns in the
Utility Matrix. (In this assignment you are given a number of ratings, from which it is
possible to build a utility matrix.) In addition to using various similarity measures for finding
the most similar items or users, one can use latent factor models (matrix decomposition)
and other hybrid approaches to improve on the training and test data RMSE scores.
The goal of this assignment is to allow you to develop collaborative filtering models that
can predict the rating of a specific item from a specific user given a history of other
ratings.


To evaluate the performance of your results we will use the Root-Mean-Squared-Error
(RMSE).


## Data Description:
The training dataset consists of 85724 ratings and the test dataset consists of 2154 ratings.
We provide you with the training data ratings and the test ratings are held out. The data are
provided as text in train.dat and test.dat, which should be processed appropriately.
train.dat: Training set (UserID <tab separator> ItemID <tab separator> Rating (Integers 1
to 5) <tab separator> Timestamp (Unix time stamp).
test.dat: Testing set (UserID <tab separator> ItemID, no rating provided).
format.dat: A sample submission with 2154 ratings being all 1 (The one val
