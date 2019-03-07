import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating

# Define function 
def loadAnimeNames():
    animeNames = {}
    with open("anime_cleaned.csv", encoding='ascii', errors="ignore") as f:
        for line in f:
            fields = line.split(',')
            animeNames[int(fields[0])] = fields[1]
    return animeNames

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
print("\nLoading anime names...")
nameDict = loadAnimeNames()
 
lines = sc.textFile("animelists_als.csv")
parsedlines = lines.map(parseline)
ratings = parsedlines.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()


# Build the recommendation model using Alternating Least Squares
print("\nTraining recommendation model...")
rank = 10
numIterations = 20
model = ALS.train(ratings, rank, numIterations)

userID = int(sys.argv[1])

#print("\nRatings for user ID " + str(userID) + ":")
#userRatings = ratings.filter(lambda l: l[0] == userID)
#for rating in userRatings.collect():
#    print (nameDict[int(rating[1])] + ": " + str(rating[2]))

print("\nTop 10 recommendations:")
recommendations = model.recommendProducts(userID, 10)
for recommendation in recommendations:
    print (nameDict[int(recommendation[1])] + \
        " score " + str(recommendation[2]))

