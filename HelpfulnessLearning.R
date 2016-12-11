library(RSQLite)
library(tm)
library(wordcloud)
library(RSQLite)
library(data.table)
library(genderizeR)
library(qdap)

# DB read
db <- dbConnect(dbDriver("SQLite"), "database.sqlite")

df <- dbGetQuery(db, "
                      SELECT *
                      FROM Reviews
                      WHERE HelpfulnessDenominator >= 10")

# reviews <- data.table(reviews,keep.rownames=TRUE)

head(reviews, 1)
str(reviews)

df$hratio <- df$HelpfulnessNumerator/df$HelpfulnessDenominator
df$wordCount <- sapply(df$Text, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))

reviews$gender <- findGivenNames(reviews$ProfileName, textPrepare = FALSE)

# text2vec

library(text2vec)
library(data.table)
# data("movie_review")
setDT(reviews)
setkey(reviews, Id)
# set.seed(2016L)
all_ids = reviews$Id
train_ids = sample(all_ids, 4000)
test_ids = setdiff(all_ids, train_ids)
train = reviews[J(train_ids)]
test = reviews[J(test_ids)]

vocablist <- data.frame(vocab)


# define preprocessing function and tokenization fucntion
prep_fun = tolower
tok_fun = word_tokenizer
  
it_train = itoken(train$Text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$Id, 
                  progressbar = FALSE)
vocablist <- create_vocabulary(it_train)
vocab

vectorizer = vocab_vectorizer(vocab)
t1 = Sys.time()
dtm_train = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

dim(dtm_train)
dim(train$Id)

print(train$Id)
print(rownames(dtm_train))

identical(rownames(dtm_train), train$Id)

library(glmnet)
NFOLDS = 4
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 5-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))

print(vocab)
