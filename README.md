# Kaggle_CatsVsDogs

These scripts were made whilst learning to use the Bag of Words model of OpenCV on the Cats Vs Dogs Kaggle dataset. The data from the competition can be found at https://www.kaggle.com/c/dogs-vs-cats/data


Tips:
- Create a directory named "Cache".
- Download the training and test zip files from https://www.kaggle.com/c/dogs-vs-cats/data and unzip them into folders named "train" and "test1".

You can see how well tweaking parameters work without submissions to Kaggle. For this run "createFeatures.py" first, followed by "trainFeatures.py".


To submit to Kaggle, run "predictTestSet.py". Untweaked, this code achieves a score of 0.58023 on the leaderboard.
