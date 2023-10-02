Predicting Customer Churn

Scenario: A marketing agency has many customers who use their service to produce ads for customer websites. They've noticed that they have quite a bit of churn in clients. They randomly assign account managers but want to be more strategic by assigning their best account managers to the clients most at risk to churn. They have provided their historical client data, so that we can leverage to build a machine-learning model to quickly identify at-risk customers.

Data Source: https://www.kaggle.com/datasets/hassanamin/customer-churn

Key Data Points
•	Name: Name of the latest contact at Company
•	Age: Customer Age
•	Total Purchase: Total Ads Purchased
•	Account Manager: Binary 0=No manager, 1= Account manager assigned
•	Years: Total Years as a customer
•	Number of Websites Used: Number of websites that use the service.
•	Onboard Date: The date that the name of the latest contact was onboarded
•	Location: Client HQ Address
•	Company: Name of Client Company
•	Churn: Yes (1) or No (0)

For data exploration we employed multiple confusion matrix to determine which will produce the best outcome.  These included Logistic Regression, Decision Tree, K-Nearest Neighbors Classifier, Random Forest, Gradient Boosting Classifier, Ada Boost Classifier and Extra Tree Classifier.  Ultimately it appears that that Random Forest provided the best overall performance, while Ada Boost Classifier proved to be the best option for mean and standard deviation of the accuracy scores.

Additionally, we used multiple set up for Neural Networks in order to attempt to build predictions based on Machine Learning.  Each model produced different outcomes or course based on the number of Neurons and Layers employed.  Ultimately we were able to achieve a predictive accuracy score of 87% within our designated 100 Epochs by utilizing a setup consisting of the initial two layers, the sigmoid activation function for each layer, and 500 & 300 neurons respectively
