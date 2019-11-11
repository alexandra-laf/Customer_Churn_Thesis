library(glmnet)
#install.packages("tidyr")
library(tidyr)
#install.packages("survival")
library(survival)
#install.packages("readxl")
library(readxl)
#install.packages("tm")
library(tm)
#install.packages("stringr")
library(stringr)
#install.packages("miscTools")
library(miscTools)
#install.packages("ibmdbR")
library(ibmdbR)
#install.packages("mclust")
library(mclust)
#install.packages("cluster")
library(cluster)
#install.packages("factoextra")
library(factoextra)
#install.packages("creditmodel")
library(creditmodel)
#install.packages("backports")
library(backports)
#install.packages("tidyverse")
library(tidyverse)
#install.packages("caret")
library(caret)
#install.packages("MLmetrics")
library(MLmetrics)
#install.packages("MASS")
library(MASS)
#install.packages("ResourceSelection")
library(ResourceSelection)
#install.packages("randomForest")
library(randomForest)
library(rpart.plot)
#install.packages("imbalance")
library(imbalance)
library(dplyr)
library(ggplot2)
library(scales)
library(e1071)
theme_set(theme_bw())

#read the data (exported dataframe as csv file from jupyter notebook)
datamou=read.csv("C:/Users/user/Desktop/FINAL_PROJECT_NBG_CHURN/data_for_classific_median_fromubuntu.csv")
colnames(datamou)

#remove the fisrt column
datamou <- subset(datamou, select = - client_username_s )

datamou
#see the type of data columns
sapply(datamou,class)

X=datamou
#Convert that column to a factor
X$Cluster=factor(X$Cluster)
X$Target_by_med=factor(X$Target_by_med)

levels(X$Target_by_med)

###DistributioN of prodictors
hist1=hist(X$countt,col="black")
hist2=hist(X$recency,col="black")
hist3=hist(X$frequency,col="black")
hist4=hist(X$max_dist,col="black")
hist5=hist(X$average_trans_dur,col="black")
hist6=hist(X$diff_maxd_mind_days,col="black")
hist7=hist(X$start_trans_end_data,col="black")




group_colors <- c("blue","red")

p=ggplot(X, aes(x=Target_by_med,group=Cluster,fill=Target_by_med)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ),stat= "count", vjust = -.5) +
  facet_grid(~Cluster) +
  scale_fill_manual("NO CHURN - CHURN",labels = c("0","1"),values =group_colors)+
  labs(x = "NO CHURN - CHURN", y = "PERCENTAGE") + 
  ggtitle("Percentage of no churners- churners \nin each of the clusters 0 and 2")

p

#FOR THE ALL DATA SET

group_colors <- c("blue","red")

q=ggplot(X, aes(x=Target_by_med,group=1,fill=Target_by_med)) + 
  geom_bar(aes(y = ..prop..,fill = factor(..x..)), stat = "count") +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ),stat= "count", vjust = -.5) +
  facet_grid(~1)+
  scale_fill_manual("NO CHURN - CHURN",labels = c("0","1"),values = group_colors) +
  ggtitle("Percentage of no churners - churners in the whole dataset") +
  labs(x = "NO CHURN - CHURN", y = "PERCENTAGE") +
  theme(strip.background.x = element_blank(), strip.text.x = element_text(size=0))

q







#### Classification phase #########




#Train Test Split the all data

set.seed(12)
XY_train_test =train_test_split(X,split_type = "Random", prop = 0.7)
XY_train = XY_train_test$train
XY_all_test = XY_train_test$test
#train test split again for the test so as t create the development set and the test
XY_dev_test=train_test_split(XY_all_test,split_type = "Random", prop = 0.7)
XY_dev=XY_dev_test$train
XY_test=XY_dev_test$test

#Separate the target from the data

Y_train=XY_train$Target_by_med
Y_dev=XY_dev$Target_by_med
Y_test=XY_test$Target_by_med

# Separate the X data 
myvars=c("countt", "recency", "frequency","max_dist","average_trans_dur","diff_maxd_mind_days","start_trans_end_data","Cluster")
X_train=XY_train[myvars]
X_dev=XY_dev[myvars]
X_test=XY_test[myvars]


###########################     LOGISTC   REGRESSION  (Multiple logistic regression      )####################################


##############  1st approach  ###########################


# Fit the model
model1 <- glm(Target_by_med ~., data =XY_train, family = binomial)



# Summarize the model
summary(model1)
#Error "not defined because of singularities" will occur due to strong correlation between your independent variables

summary(model1)$coef

#I dont put the start_trans_end_data
# Fit the model
modelnew <- glm(Target_by_med ~countt+recency+frequency+max_dist+average_trans_dur+diff_maxd_mind_days+Cluster, data =XY_train, family = binomial)
summary(modelnew)
summary(modelnew)$coef


###predictions

X_dev_NB=X_dev # for naive bayes
X_dev=subset(X_dev, select = -start_trans_end_data)#exclude the parameter start_trans_end_data



#for each client predict the probability
probabilities <- modelnew %>% predict(X_dev, type = "response")

predicted_classes <-ifelse(probabilities > 0.5,1,0)
head(predicted_classes)

#assess model accuracy
mean(predicted_classes == Y_dev)
f1score=F1_Score(Y_dev,predicted_classes, positive = 1)
f1score
#confusion matrix
xtab=table(predicted_classes,Y_dev)
conf1=confusionMatrix(xtab)
conf1
#ARI
mc_ari=adjustedRandIndex(predicted_classes,Y_dev)
mc_ari 

library(broom)
glance(modelnew)
tidy(modelnew)
R_pseudo2=1-(184094/557300) 
R_pseudo2
###########################    1b)  approach  LOGISTC   REGRESSION  (Multiple logistic regression  )####################################

####    feature selection

step(model1,direction='both',k=log(nrow(XY_train)))


#Αυτο μου καταλήγει στο μοντέλο που πηρα πριν το τελικό το modelnew




############# 2nd approach WORK WITH THE  LOG all  DATA   #########################


#######   the data are very skew    ####################


# create the log +1 
#Make sure that the predictor variables are normally distributed. If not, you can use log, root, Box-Cox transformation

Xforlog=X
#columns+1
Xforlog$countt=Xforlog$countt+1
Xforlog$recency=Xforlog$recency+1
Xforlog$frequency=Xforlog$frequency+1
Xforlog$max_dist=Xforlog$max_dist+1
Xforlog$average_trans_dur=Xforlog$average_trans_dur+1
Xforlog$diff_maxd_mind_days=Xforlog$diff_maxd_mind_days+1
Xforlog$start_trans_end_data=Xforlog$start_trans_end_data+1

#take the natural log 
Xforlog[,c(1:7)]=log(Xforlog[,c(1:7)])
Xforlog

###DistributioN of new prodictors
hist8=hist(Xforlog$countt,col="black")
hist9=hist(Xforlog$recency,col="black")
hist10=hist(Xforlog$frequency,col="black")
hist11=hist(Xforlog$max_dist,col="black")
hist12=hist(Xforlog$average_trans_dur,col="black")
hist13=hist(Xforlog$diff_maxd_mind_days,col="black")
hist14=hist(Xforlog$start_trans_end_data,col="black")



#Train Test Split the all data
set.seed(12)
XY_train_test_log =train_test_split(Xforlog,split_type = "Random", prop = 0.7)
XY_train_log = XY_train_test_log$train
XY_all_test_log = XY_train_test_log$test
#train test split again for the test so as to create the development set and the test
XY_dev_test_log=train_test_split(XY_all_test_log,split_type = "Random", prop = 0.7)
XY_dev_log=XY_dev_test_log$train
XY_test_log=XY_dev_test_log$test


#Separate the target from the data

Y_train_log=XY_train_log$Target_by_med
Y_dev_log=XY_dev_log$Target_by_med
Y_test_log=XY_test_log$Target_by_med

# Separate the X data 
myvars=c("countt", "recency", "frequency","max_dist","average_trans_dur","diff_maxd_mind_days","start_trans_end_data","Cluster")
X_train_log=XY_train_log[myvars]
X_dev_log=XY_dev_log[myvars]
X_test_log=XY_test_log[myvars]


# Fit the model
model1_log <- glm(Target_by_med ~., data =XY_train_log, family = binomial)

# Summarize the model
summary(model1_log)
summary(model1_log)$coef


#for each client predict the probability
probabilities_4<- model1_log %>% predict(X_dev_log, type = "response")

predicted_classes_4 <-ifelse(probabilities_4 > 0.5,1,0)
head(predicted_classes_4)

#assess model accuracy
mean(predicted_classes_4 ==Y_dev_log)
f1score_4=F1_Score(Y_dev_log,predicted_classes_4, positive = 1)
f1score_4
#confusion matrix
xtab_4=table(predicted_classes_4,Y_dev_log)
conf_4=confusionMatrix(xtab_4)
conf_4
#ARI
mc_ari_4=adjustedRandIndex(predicted_classes_4,Y_dev_log)
mc_ari_4 

R_pseudo2log=1-(204358/557300)
R_pseudo2log

############################  4rth dokimh  ##########################


#feature selection on log

step(model1_log,direction='both',k=log(nrow(XY_train_log)))

#none variable extracted


###########################################   CLASSIFIER 2nd  ################################

# Create a Random Forest model with default parameters
model_random_for <- randomForest(Target_by_med ~ ., data = XY_train, importance = TRUE)
model_random_for   ## memory error 
############################################################################



#XY_train$Target_by_med=as.factor(XY_train$Target_by_med)
#X$Target_by_med=as.factor(X$Target_by_med)
#XPERC=X %>% group_by(Target_by_med) %>% summarize(Target_by_med_count = n())





################################  Νaive bayes  #########################




model_NB = naiveBayes(Target_by_med ~., data =XY_train)

#predictions
estimt_NB_classes=predict(model_NB,X_dev_NB,type="class")
estimt_NB_prob=predict(model_NB,X_dev_NB,type="raw")



#assess model accuracy
mean(estimt_NB_classes ==Y_dev)
#F1 SCORE
f1score_NB=F1_Score(Y_dev,estimt_NB_classes, positive = 1)
f1score_NB
#confusion matrix
xtab_NB=table(estimt_NB_classes,Y_dev)
conf_NB=confusionMatrix(xtab_NB)
conf_NB
#ARI
mc_ari_NB=adjustedRandIndex(estimt_NB_classes,Y_dev)
mc_ari_NB 




###CONCLUSIONS ####




###################LOGI PERFORMS BETTER #######

######################## TEST SET ###########################3

###predictions

X_test=subset(X_test, select = -start_trans_end_data)


#for each client predict the probability
probabilities_test <- modelnew %>% predict(X_test, type = "response")

predicted_classes_test <-ifelse(probabilities_test > 0.5,1,0)


#assess model accuracy
mean(predicted_classes_test == Y_test)
##f1 score
f1score_test=F1_Score(Y_test,predicted_classes_test, positive = 1)
f1score_test
#confusion matrix
xtab_test=table(predicted_classes_test,Y_test)
conf_matrix_test=confusionMatrix(xtab_test)
conf_matrix_test
#ARI
mc_ari_test=adjustedRandIndex(predicted_classes_test,Y_test)
mc_ari_test 


probabilities_test

##extract
write.csv(probabilities_test,'C:\\Users\\user\\Desktop\\FINAL_PROJECT_NBG_CHURN\\probabilities_test.csv', row.names = FALSE)


##########################################














