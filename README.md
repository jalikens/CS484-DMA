# CS484-DMA
Repository for Final Project for CS484
Team: The Data Mining Administration (DMA)
Members: John Likens, O Dodart, Matthew Browne

Project Report

Introduction
Background: Our project focused on credit card fraud and addressing how to handle that problem. Credit card fraud is an issue that can negatively impact people worldwide. While there may be models capable of detecting fraud, we wanted to tackle the problem ourselves. We wanted to discover what factors could lead to credit card fraud in order to further prevent credit card fraud from occurring.

Motivation: In order to do this, our goal was to take a large dataset of transactions to build an optimal model capable of identifying fraud. By using this data model, we can analyze the results of our models and determine what common factors and patterns lead to fraudulent transactions. Some of the various factors that may determine if a situation was fraud may include the location, how big the purchase, what was the purchase, time of day, etc.  

Challenges: A major problem we ran into pretty early on was finding the right dataset. The first major issue we discovered was that many datasets of transactional data were predominantly skewed towards non-fraudulent data. This would obviously lead to an inaccurate model, and our results would not be very impactful. Another major issue was the information given in some of the datasets; in some cases, we found that there was not enough entries to build a strong model, in other datasets there was unnecessary information given to us such as the seller’s name or there were other features considered that weren’t originally in the dataset, such as the customer’s age. 

After numerous datasets, we found a dataset that contained 339607 transactions with 15 features (['trans_date_trans_time', 'merchant', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'merch_lat', 'merch_long']), such as transaction time, merchant, and location of transaction, and 1 label ([‘is_fraud’], 0 = not, 1 = fraud). While this dataset did not contain every feature we needed to analyze, it did contain features we could use to create those needed features. For example, with ‘dob’ (date-of-birth) and the date of the transaction, we could calculate a customer’s age at the time of purchase. With this dataset established we could conduct our experiment.

Summarization of Contributions

Matthew Browne
- Contributed to Written Reports
- Contributed to Final Presentation 

O Dodart
- Contributed to Written Reports
- Contributed to Hypertuned Parameters

John Likens
- Contributed to Model Creation
- Contributed to Data Charts Notebook

Method:
Problem 1: Skewness in Data
In order to handle the issue of skewness in data, we performed undersampling, where we ensure the number of samples of the majority label class (in our case, not fraud) is equal to the number of samples in the minority label class (fraud). This removes the skewness in data and allows us to build a more accurate model.

Problem 2: Building Models
KNN & PCA: By building a KNN model, we are able to predict whether a transaction was fraud or not by assigning it based on its majority of k closest neighbors in the model. In order to train the model, we split the data randomly into DTrain and DTest. For each sample in DTrain, we find their k closest neighbors, and assign it a class, c based on the majority those neighbors were assigned (vc). This will assign a class to each sample in DTrain, and we can evaluate our model using DTest by calculating the accuracy and F1-score (precision/recall).

Using PCA, we can take our original dataset xn and condense it to a projected dataset yn with less dimensions (since we started with 914 originally), such that yn = UTxn, where U is a subspace within the domain RDxM. This will allow us to maximize the variance of our transformed data with minimal loss of information. 
 
Neural Networks (MLP): Given a set of layer sizes and an activation function yhat, we use the output s(w0 + w1x1 + w2x2 + …  = s(wTx) of layer 1 L1 as an input for the next layer, which will continue to feed-forward to hidden layers build a single output layer.

Logistic Regression Model: By building a model based of the binary classification system (y = {0,1}), we can apply the Bernoulli Distribution to train the model to predict transactional data. This is done such that the probability of an input x being in the positive class (in this case, fraud) being p(y=1|x,w, where w is weights within the model such that w ∈ RD+1 ), and 0 otherwise.

Random Forests: This model takes in a random subset from DTrain to build n decision trees, and collects the votes from those different decision trees to predict whether or not a transaction was fraud or not.


Experiment Details

Data Metrics:
['trans_date_trans_time', 'merchant', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'merch_lat', 'merch_long']

trans_date_trans_time → Month, Day of Week, Hour 
Merchant → Name of merchant
category → Type of purchase 
amt → Total cost of transaction (in dollars)
city and state 
lat and long → latitude and longitude of transaction
city_pop → population of city
job → Occupation merchant had 
trans_num
dob → date of birth 
merch_lat and merch_long → latitude and longitude of the merchant during the purchase

Pre-Processing
We altered the date of birth (‘dob’) metric into the pandas datetime for easier recognition. 
We added dimensions for time for hour, month, day of week for transaction, customer age at transaction, and distance of customer from transaction
We created one hot encoding for abnormal time of transactions, day of the week, state, merchant, category of purchase, and city
We dropped several dimensions after translating the dataset to machine-friendly terms or determining a dimension to be unnecessary, including the transaction number, the job of the customer, the transaction data, the state of the transaction, the city of the transaction, the name of the merchant, the category of the purchase, day of the week of the transaction, and the day of birth of the customer.

Final Data Metrics:
[‘hour_of_transaction’, ‘month_of_transaction’, ‘dow_of_transaction’, ‘cust_age’ (customer age), ‘distance_of_transaction’, ‘normal_transaction_time’ (6:00 am - 9:00 pm is considered a normal time), ‘amt’, ‘city_pop’, 


Models Used:
- Logistic Regression
- Random Forest Model
- KNN with PCA
- Neural Network (MLP)
- Heat Map

In order to address the issue of skewness, we have decided to apply undersampling and oversampling to each of the models, and compare the results of each model to determine which method worked best in removing bias, while minimizing loss in data representation and oversaturating our dataset. 

In order to calculate the correlation between features and transactions, we used a heat map to calculate the values between features. A value between a feature and ‘is_fraud’ determined how strong the correlation was in detecting fraudulent activity.

Hyper-Parameter Tuning:

Logistic Regression (Undersampling)
test size = 0.06
solver = ‘newton-cg’
penalty = ‘l2’
inverse standardization strength(c) = 1

Random Forest Classifier (Undersampling)
test size = 0.06
decision trees = 104
max depth of decision trees = 54
KNN with PCA (Undersampling)
k = 1
weight = uniform

Neural Network Classifier (Undersampling)
layers = (914,500,250,100,50,1)
activation = ‘relu’

Logistic Regression (Oversampling)
test size = 0.26
solver = ‘liblinear’
penalty = ‘l2’
inverse standardization strength(c) = 1

Random Forest Classifier (Oversampling)
test size = 0.2
decision trees = 100
max depth of decision trees = 60

KNN with PCA (Oversampling)
k = 
weight = 

Neural Network Classifier (Oversampling)
layers = 
activation = 

Evaluation Metrics Used:

Accuracy and F1-Score (Acc and Precision): In the original dataset, it would not have been feasible to use accuracy as an evaluation metric due to the imbalance in data. By undersampling and oversampling, however, we can safely use accuracy as an evaluation tool without worry of imbalance. F1-Score is a second metric we used to optimize our models. 

When tuning the hyperparameters for our models, accuracy of the test set, recall of the model, and the f-score were put into consideration. The recall of the model and the f-score of the model’s accuracy were emphasized more because of the danger of the Type II Error concerning credit card fraud, or the instance where a transaction was fraud but was not labelled as fraud. In addition, the focus on the test set for our optimization ensures the model can handle unfamiliar data, analyze the data given, and determine whether a transaction is fraudulent.

Results and Analysis
Heat Map:
The figure above is the result of the heat map calculated with all the features. The closer a value is to 1, the stronger the correlation of the two corresponding values. Using this, we can see that with a value of .2, ‘amt’, or the amount of the transaction has the strongest correlation with ‘is_fraud’, followed by normal_transaction_time (0.98), customer age (.013), and hour of transaction (0.12). These results indicate that these variables are the strongest factors in determining whether or not a transaction is fraud.

Confusion Matrices
In calculating the precision, recall and F1-score for the various models, we created the confusion matrix for each model. 

Final Results
The table below reveals the final Accuracy, F1-Score, Precision, and Recall for each model where applicable. This table will also reveal the optimal hypertune values used in each model to reach their corresponding evaluation metrics:



                      KNN & PCA          RF                   NN(MLP)             Logistic Reg.
Acc. (Under)        0.9948          1.0,0.9626            0.9812,0.8645           0.92540,8972
F1 (Under)          0.1442              N/A               0.9809,0.8571           0.9259,0.8922
Prec. (Under)       0.6122              N/A                 1.0                   0.9232
Recall (Under)      0.0817              N/A                 0.87                  0.91
Acc. (Over)
F1 (Over)
Prec. (Over)
Recall (Over)
Final Param.




As we can see from the table, we can see that in terms of accuracy, (Insert Model with final parameters) is the most optimal model in determining whether or not a credit card transaction is fraud or not. 

Conclusions and future work
To conclude our report, we sought out to learn what factors most into credit card fraud by building a model capable of accurately detecting credit card fraud using a large dataset of transactional data. One issue that impacted our results was computing power. When we were applying overfitting to our models, for example, we had to lower the size of the original oversampling size due to how much time it took to originally run. 

While our experiment was successful, there are considerations that could lead to results different from ours. In our dataset, for example, our data was taken from various cities all with different situations, and we mainly focused on the city population regarding the city itself. This leads to the question of how may these factors change if we target one specific city and build a model from there? 

Github Link & Data
Github Link
 

