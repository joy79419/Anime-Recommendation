from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import numpy as np


def parseline(line):
    fields = line.split(',')
    userid = fields[0]
    animeid = fields[1]
    score = fields[2]
    return (userid, animeid, score)

# pyspark set-up
conf = SparkConf().setMaster("local[*]").setAppName("AnimeRecommendationsALS")
sc = SparkContext(conf = conf)
sc.setCheckpointDir('checkpoint')

# Build rating object for ALS 
print("\nLoading data...")

 
lines = sc.textFile("animelists_als.csv")
parsedlines = lines.map(parseline)
ratings = parsedlines.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()
test, train = ratings.randomSplit(weights=[0.2, 0.8], seed=1)


# Build the recommendation model using Alternating Least Squares
print("\nTraining recommendation model...")
rank = 5
numIterations = 10
model = ALS.train(train, rank, numIterations)

#userID = int(sys.argv[1])
#print("\nRatings for user ID " + str(userID) + ":")
#userRatings = ratings.filter(lambda l: l[0] == userID)
#for rating in userRatings.collect():
#    print (nameDict[int(rating[1])] + ": " + str(rating[2]))

#print("\nTop 10 recommendations:")
#recommendations = model.recommendProducts(userID, 10)
#for recommendation in recommendations:
#    print (nameDict[int(recommendation[1])] + \
#        " score " + str(recommendation[2]))

print("\nCalculating Test RMSE...")
testData = test.map(lambda p: (p.user, p.product))
predictions = model.predictAll(testData).map(lambda r: ((r.user, r.product), r.rating))
ratingsTuple = test.map(lambda r: ((r.user, r.product), r.rating))
scoreAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])

metrics_rating = RegressionMetrics(scoreAndLabels)
print("\nTest RMSE = %s" % metrics_rating.rootMeanSquaredError)

print("\nCalculating NDCG@10...")
itemFactors = model.productFeatures().map(lambda factor: factor[1]).collect()
itemMatrix = np.array(itemFactors)
imBroadcast = sc.broadcast(itemMatrix)
userMovies = ratings.map(lambda rating: (rating.user,rating.product)).groupBy(lambda x:x[0])
userMovies = userMovies.map(lambda x:(x[0], [xx[1] for xx in x[1]] ))
userVector = model.userFeatures().map(lambda x:(x[0],np.array(x[1])))
userVector = userVector.map(lambda x: (x[0],imBroadcast.value.dot((np.array(x[1]).transpose()))))
userVectorId = userVector.map(lambda x : (x[0],[(xx,i) for i,xx in enumerate(x[1].tolist())]))
sortUserVectorId = userVectorId.map(lambda x:(x[0],sorted(x[1],key=lambda x:x[0],reverse=True)))
sortUserVectorRecId = sortUserVectorId.map(lambda x: (x[0],[xx[1] for xx in x[1]]))
sortedLabels = sortUserVectorRecId.join(userMovies).map(lambda x:(x[1][0],x[1][1]))

metrics_rank = RankingMetrics(sortedLabels)
print("\nNDCG@10 = %s" % metrics_rank.ndcgAt(10))


