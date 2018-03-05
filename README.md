![](/images/kaggle_logo.png)

*https://www.kaggle.com/*

**The goal of this repository is to get more experience in processes of data-analysis: data collection, preprocessing, cleaning, plotting, feature engineering, etc. Also would be nice to improve my R and Python skills as well as become more comfortable with mostly used machine learning algorithms. Here I'm going to replicate works of others and try out my own ideas.**

---

# [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic#description)

<img align="left" src="/images/titanic_small.png">

*The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this challenge, the goal is to complete the analysis of what sorts of people were likely to survive and apply the tools of machine learning to predict which passengers survived the tragedy. [Data here.](https://www.kaggle.com/c/titanic/data)*

For this challenge I chose a popular [Python notebook](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook) from Kaggle kernels and replicated it in R. During the process I decided to slightly change some details and order of operations, but the essence remains the same: all continuous numerical features (*Age*, *Fare*) are split into categories by values or distribution, fully prepared data contains only discrete numerical values, no one-hot encoding, not used *Cabin* feature with many empty values, used different classifiers, but no hyper-parameter tuning (well, in my case *caret* tries some different hyper-parameters automatically).

[**My script in R**](/titanic/titanic.R)

**Current test set accuracy results:**
* Logistic Regression - 0.73205
* Support Vector Machines with Radial Basis Kernel - 0.76555
* k-Nearest Neighbors - 0.76555
* Gaussian Naive Bayes - 0.71770
* Perceptron - 0.74162
* Support Vector Machines with Linear Kernel - 0.76555
* Stochastic Gradient Descent (mlp) - 0.62679
* Decision Tree - 0.76555
* Random Forest - 0.77033

---

# [Telecom company data analysis and churn prediction](https://www.kaggle.com/hkalsi/telecom-company-customer-churn)

<img align="left" src="/images/churn_small.png">

*The churn rate, also known as the rate of attrition, is the percentage of subscribers to a service who discontinue their subscriptions to that service within a given time period. For a company to expand its clientele, its growth rate, as measured by the number of new customers, must exceed its churn rate. The rate is generally expressed as a percentage. [Data here.](https://www.kaggle.com/hkalsi/telecom-company-customer-churn/data)*

This is my final data anlysis project for Tele2 Big Data Academy 2018. For this task I chose a telecom company customers dataset with labeled churn cases and applied known methods of data cleaning, feature engineering and exploratory analysis. After undertanding main trends and criteria leading to customer churn, I tried some popular machine learning algorithms to predict potential churners. Modeling results in test set could be seen below. 

[**My script in R**](/churn-prediction)

<img src="/images/example1.png" height="480"/><img src="/Data-Science-Specialization-Coursera/Course-4-Exploratory-Data-Analysis/Course-Project-1/plot2.png" height="210"/><img src="/Data-Science-Specialization-Coursera/Course-4-Exploratory-Data-Analysis/Course-Project-1/plot3.png" height="210"/><img src="/Data-Science-Specialization-Coursera/Course-4-Exploratory-Data-Analysis/Course-Project-1/plot4.png" height="210"/>



