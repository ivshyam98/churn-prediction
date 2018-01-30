## Kaggle - The Home of Data Science & Machine Learning
![](/images/kaggle_logo.png)

*https://www.kaggle.com/*

**The goal of this repository is to get more experience in processes of data-analysis: data collection, preprocessing, cleaning, plotting, feature engineering, etc. Also would be nice to improve my R and Python skills as well as become more comfortable with mostly used machine learning algorithms. Here I'm going to replicate works of others and try out my own ideas.**

# [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic#description)

<img align="left" src="/images/titanic_small.png">

*The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this challenge, the goal is to complete the analysis of what sorts of people were likely to survive and apply the tools of machine learning to predict which passengers survived the tragedy.* 
[**Data source**](https://www.kaggle.com/c/titanic/data)

For this challenge I chose a popular [**Python notebook**](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook) from Kaggle kernels and replicated it in R. During the process I decided to slightly change some details and order of operations, but the essence remains the same: all continuous numerical features (*Age*, *Fare*) split into categories by values or distribution, fully prepared data contains only discrete numerical values, no one-hot encoding, not used *Cabin* feature with many empty values, used different classifiers, but no hyper-parameter tuning (well, in my case *caret* tries some different hyper-parameters automatically).

**My script in R**
