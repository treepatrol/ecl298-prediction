# load packages
library(tidyverse)
library(randomForest)

# load data
yose_plots <- read_csv("data/IRMA_YOSE/YOSE_SugarPine_Event.csv")
yose_trees <- read_csv("data/IRMA_YOSE/YOSE_SugarPine_Occurrence.csv")
# look at the variables
colnames(yose_plots)
colnames(yose_trees)

# data cleaning
# remove irrelevant event-level columns
yose_plots <- yose_plots %>% select(eventID, samplingProtocol, sampleSizeValue_total, startDateTime, verbatimLatitude, verbatimLongitude, eventRemarks)
# remove irrelevant occurrence-level columns
yose_trees <- yose_trees %>% select(-basisOfRecord, -scientificName, -individualCount, -organismQuantity, -organismQuantityType, -publicDisplay, -dataAccess, -lifeStage, -sex, -behavior, -covariateSample, -preparations, -identifiedBy, -identificationReferences, -identificationRemarks, -identificationQualifier, -identificationVerificationStatus, -materialSampleID, -recordNumber, -organismRemarks, -identificationID)

# join to add plot info to the tree table 
yose <- left_join(yose_trees, yose_plots, by = "eventID")

# remove "plot" 48 since it doesn't meet plot criteria
# check that these 34 trees are from plot 48
yose48 <- yose[yose$eventID == 48, ]
# subset to keep only trees in normal plots
yose <- yose[yose_trees$eventID != 48, ]


# remove columns that aren't predictors
yose_clean <- yose %>% select(percentLive, diameter, height, pitchTubes, exitHoles, activeBranchCanker, inactiveBranchCanker, activeBoleCanker, inactiveBoleCanker, deadTop, boleChar_text, damageCodes)

# use only complete cases for now
yose_clean <- yose_clean[complete.cases(yose_clean), ]

# split data in training (700 observations) and testing (300 observations) 
i <- sample(960)[1:700]
train <- yose_clean[i, ]
test  <- yose_clean[-i, ]

# fit a random forest model without tuning
rf <- randomForest::randomForest(train$percentLive~., data=train,  importance=TRUE)

# quick look at variable importance
varImpPlot(rf)

# damageCodes emerged as the most important predictor of percent live, which is interesting since this was compiled from notes in an attempt to make our field observations consistent with USFS standardized damage codes. The beetle-related variable number of exit holes was the next most importatnt, but the other beetle-variable pitch tubes was not so important. Blister rust signs (various types of cankers) were the least important in predicting overall canopy health as assessed by percent live. 
