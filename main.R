# Project: Telecom company customer churn prediction 
# Author: Karolis Matuliauskas
# Data: https://www.kaggle.com/hkalsi/telecom-company-customer-churn



#   TABLE OF CONTENT:
#   - Getting and cleaning data;
#   - Feature engineering and selection;
#   - Exploratory data analysis;
#   - Simple modeling;



##########
########## GETTING AND CLEANING DATA
##########


# Load libraries
library(plyr)
library(lubridate)
library(tidyverse)
library(data.table)
library(caret)
library(ggalluvial)
library(onehot)
library(pROC)
library(gridExtra)
set.seed(1)


### Read data 
na.values <- c('', '?', 'MISSINGVAL', 'NA')  # values from data documentation
train_acc <- fread('data/Train_AccountInfo.csv', na.strings=na.values)
train_dem <- fread('data/Train_Demographics.csv', na.strings=na.values)
train_ser <- fread('data/Train_ServicesOptedFor.csv', na.strings=na.values)
train_churn <- fread('data/Train.csv', na.strings=na.values)


# Check if data suitable to join
str(train_acc)
str(train_dem)
str(train_ser)  # different number of rows - services gathered to 1 column
str(train_churn)


# Spread different services into separate columns
train_ser <- spread(train_ser, TypeOfService, SeviceDetails)


# Merge data to one table and order by CustomerID
full_data <- train_acc %>% 
    left_join(train_dem, by=c('CustomerID'='HouseholdID')) %>%
    left_join(train_ser, by='CustomerID') %>%
    left_join(train_churn, by='CustomerID') %>%
    arrange(CustomerID)


# Preview data 
str(full_data)
glimpse(full_data)
summary(full_data)
head(full_data)
as.data.frame(x=colSums(is.na(full_data)))   # NA values in: TotalCharges, ContractType, Country, State, Education, Gender


# clean data by every feature
clean_data <- full_data %>%
    # unuseful feature
    select(-CustomerID) %>% 
    # convert to date (Date of data collection - same in all observations)
    mutate(DOC = dmy(DOC)) %>% 
    # drop observations with NA (new customers - didn't pay yet)
    filter(!is.na(TotalCharges)) %>%  
    # conver to numeric
    mutate(TotalCharges = as.numeric(TotalCharges)) %>%  
    # convert to date (Date of entry as customer)
    mutate(DOE = dmy(DOE)) %>%    
    # drop observations with NA (unknown contract type)
    filter(!is.na(ContractType)) %>%  
    # unuseful feature (same value in all observations)
    select(-Country) %>%   
    # unuseful feature (same value in all observations)
    select(-State) %>%     
    # standartize values
    mutate(Retired = mapvalues(Retired, from=c(0, 1), to=c('No', 'Yes'))) %>%  
    # standartize values
    mutate(HasPartner = mapvalues(HasPartner, from=c(2, 1), to=c('No', 'Yes'))) %>%  
    # standartize values
    mutate(HasDependents = mapvalues(HasDependents, from=c(2, 1), to=c('No', 'Yes'))) %>%  
    # fill NA values with 'Other' value from data set
    mutate(Education = replace_na(Education, 'Other')) %>%   
    # fill NA values with 'Male' 
    mutate(Gender = replace_na(Gender, 'Male')) %>%   
    # standartize values
    mutate(DeviceProtection = mapvalues(DeviceProtection, from='No internet service', to='No')) %>%
    # standartize values
    mutate(HasPhoneService = mapvalues(HasPhoneService, from=c(0, 1), to=c('No', 'Yes'))) %>%
    # standartize values
    mutate(MultipleLines = mapvalues(MultipleLines, from='No phone service', to='No')) %>%
    # standartize values
    mutate(OnlineBackup = mapvalues(OnlineBackup, from='No internet service', to='No')) %>%
    # standartize values
    mutate(OnlineSecurity = mapvalues(OnlineSecurity, from='No internet service', to='No')) %>%
    # standartize values
    mutate(StreamingMovies = mapvalues(StreamingMovies, from='No internet service', to='No')) %>%
    # standartize values
    mutate(StreamingTelevision = mapvalues(StreamingTelevision, from='No internet service', to='No')) %>%
    # standartize values
    mutate(TechnicalSupport = mapvalues(TechnicalSupport, from='No internet service', to='No'))


# Preview data again
str(clean_data)
colSums(is.na(clean_data))  # no NA values



##########
########## FEATURE ENGINEERING AND SELECTION
##########



clean_data <- clean_data %>%
    # create new feature - months since entry as customer
    mutate(MonthsSinceEntry = as.numeric((DOC - DOE) / 30)) %>%
    # drop Date of data entry and data collection - not usefull anymore
    select(-DOE, -DOC) %>%
    # create new feature - average customer month charge compared to base charge
    mutate(PaymentLevel = TotalCharges / MonthsSinceEntry / BaseCharges - 1) %>%
    # create new feature - service type: Phone, Interner, Phone and Interner
    mutate(ServiceType = ifelse(HasPhoneService=='Yes' & InternetServiceCategory != 'No', 'Phone and Internet', 
                         ifelse(HasPhoneService=='Yes' & InternetServiceCategory == 'No', 'Phone', 
                         ifelse(HasPhoneService=='No' & InternetServiceCategory != 'No', 'Internet', 'None')))) %>%
    # remove unusefull variable
    select(-HasPhoneService) %>%
    # create new feature - count of services
    mutate(AdditionalServices = (DeviceProtection == 'Yes') + 
                                (MultipleLines=='Yes') + 
                                (OnlineBackup=='Yes') + 
                                (OnlineSecurity=='Yes') + 
                                (StreamingMovies=='Yes') + 
                                (StreamingTelevision=='Yes') + 
                                (TechnicalSupport=='Yes') + 
                                (OnlineBackup=='Yes'))

# Drop extreme outliers
clean_data <- clean_data %>% 
    filter(PaymentLevel < 0.4 & PaymentLevel > -0.4) 

# Character variables convert to factor variables
clean_data[] = lapply(clean_data, function(x) {if (class(x)=="character") x=factor(x) else (x)})

summary(clean_data)
str(clean_data)



##########
########## EXPLORATORY DATA ANALYSIS 
##########    



# Correct variable names
var_names <- c(
    'ContractType'="Contract type",
    'ServiceType'="Service type",
    'Gender'="Gender",
    'Churn'="Churn",
    'ElectronicBilling'='Electronic billing')

# Customers overview
clean_data %>%
    select(ContractType, ServiceType, ElectronicBilling, Churn) %>%
    gather(features, values) %>%
    group_by(features, values) %>%
    dplyr::summarise(count=n()/nrow(clean_data) * 100) %>%
    ggplot(aes(x=values, y=count)) + 
    geom_bar(stat='identity', position='dodge', fill='grey') + 
    facet_wrap(~features, scales="free_x", labeller=as_labeller(var_names)) +
    geom_text(aes(x = values, y=count, label=paste(round(count), '%'), vjust = -1)) + 
    ylim(0, 90) + 
    ggthemes::theme_few() +
    theme(axis.title=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()) +
    ggtitle('Customers overview') 

# Basecharges boxplot - outliers
clean_data %>%
    ggplot(aes(x=ServiceType, y=BaseCharges)) + 
    geom_boxplot(fill='grey') + 
    facet_wrap(~ContractType) + 
    ggthemes::theme_few() +
    ylab('Base charges') + 
    ggtitle(label='Base charges distribution', 
            subtitle='By: Contract type, Service type') + 
    theme(axis.title.x=element_blank()) 

# PaymentLevel boxplot - outliers
clean_data %>%
    ggplot(aes(x=ServiceType, y=PaymentLevel)) + 
    geom_boxplot(fill='steelblue') + 
    facet_wrap(~ContractType) + 
    ggthemes::theme_few() +
    ylab('Base charges') + 
    ggtitle(label='Payment level distribution', 
            subtitle='By: Contract type, Service type') + 
    theme(axis.title.x=element_blank()) 

# Churn by contract type and base charges
ggplot(clean_data, aes(x=BaseCharges, fill=Churn)) + 
    geom_histogram(binwidth=5, position = 'stack') + 
    facet_wrap(~ContractType) + 
    labs(title='Customers churn',
         subtitle='By: Contract type, Base charges',
         x='Base charges', 
         y='Customers') + 
    ggthemes::theme_few() +
    scale_fill_brewer(palette="Set1")

# Churn by electronic billing and base charges
ggplot(clean_data, aes(x=BaseCharges, fill=Churn)) + 
    geom_histogram(binwidth=5, position = 'stack') + 
    facet_wrap(~ElectronicBilling) + 
    labs(title='Customers churn', 
         subtitle='By: Billing type, Base charges',
         x='Base charges', 
         y='Customers') +
    scale_fill_brewer(palette="Set1")+ 
    ggthemes::theme_few()

# Churn by payment method and base charges
ggplot(clean_data, aes(x=BaseCharges, fill=Churn)) + 
    geom_histogram(binwidth=5, position = 'stack') + 
    facet_wrap(~PaymentMethod) + 
    labs(title='Customers churn', 
         subtitle='By: Payment method, Base charges',
         x='Base charges', 
         y='Customers') + 
    scale_fill_brewer(palette="Set1") + 
    ggthemes::theme_few()

# Customers and churn by months since entry
ggplot(clean_data, aes(x=MonthsSinceEntry, fill=Churn)) + 
    geom_histogram(binwidth = 1) +
    facet_wrap(~ServiceType) + 
    labs(title='Customers churn', 
         subtitle='By: Service type, Months since entry',
         x='Months since entry', 
         y='Customers') + 
    scale_fill_brewer(palette="Set1") + 
    ggthemes::theme_few()

# Average customer charges by service type
clean_data %>% 
    group_by(ServiceType) %>%
    sample_n(500) %>%
    ggplot(aes(x=PaymentLevel, y=BaseCharges, color=Churn)) + 
    geom_point(alpha=0.8, size=2) + 
    facet_wrap(~ServiceType, scales="free") + 
    labs(title='Customers churn', 
         subtitle='By: Service type, Base charges, Payment level\nNote 1: Every chart shows sample of 500 customers from every service type.\nNote 2: Scales are different for every chart',
         x='Payment level', 
         y='Base charges') + 
    scale_color_brewer(palette="Set1") + 
    ggthemes::theme_few()

# Churn by education and months since entry
ggplot(clean_data, aes(x=Education, y=MonthsSinceEntry, color=Churn)) + 
    geom_jitter(alpha=0.8, size=2) + 
    labs(title='Customers churn', 
         subtitle='By: Education, Months since entry',
         x='Education', 
         y='Months since entry') + 
    scale_color_brewer(palette="Set1") + 
    ggthemes::theme_few()

# Customers by additional services, months since entry and contract type
ggplot(clean_data, aes(x=AdditionalServices, y=MonthsSinceEntry, col=Churn)) + 
    geom_jitter(width=0.4, alpha=0.8, size=2) +
    facet_wrap(~ContractType) + 
    labs(title='Customers churn', 
         subtitle='By: Additional services, Months since entry, Contract type',
         x='Additional services', 
         y='Months since entry') + 
    scale_color_brewer(palette="Set1") + 
    ggthemes::theme_few()

### Churn by other features
clean_data %>% 
    group_by(Churn, ServiceType, ContractType, ElectronicBilling) %>%
    dplyr::summarise(Freq=n()) %>%
    ggplot(aes(weight = Freq, axis1 = ElectronicBilling,
                          axis2 = ContractType, 
                          axis3 = ServiceType, 
                          axis4 = Churn )) +
    geom_alluvium(aes(fill = Churn), width = 0, knot.pos = 0.2, reverse = FALSE) +
    guides(fill = FALSE) +
    geom_stratum(width = 0.125, reverse = F) +
    geom_text(stat = "stratum", label.strata = TRUE, reverse = FALSE) +
    scale_x_continuous(breaks = 1:4, labels = c( 'Electronic billing', "Contract type", "Service type", "Churn")) +
    coord_flip() +
    labs(title='Customers churn', 
         subtitle='By: Service type, Contract type, Billing type') + 
    scale_fill_brewer(palette="Set1") + 
    ggthemes::theme_few() + 
    theme(axis.text.x=element_blank(),
          axis.ticks.x=element_blank())

# Find target segment - churn conclusion
selected <- clean_data %>%
    filter(ContractType=='Month-to-month',
           ServiceType=='Phone and Internet',
           ElectronicBilling=='Yes',
           PaymentMethod=='Electronic check',
           MonthsSinceEntry < 36,
           BaseCharges > 250) %>%
    group_by(Churn) %>%
    dplyr::summarize(Count=n()) %>%
    mutate(Percentage=paste(round(Count/sum(Count) * 100, 2), '%'))

grid.table(selected, rows = NULL)


##########
########## MODELING
##########   


# Remove features
# clean_data <- clean_data %>%
#     select(-TotalCharges)

# Load helper functions
source('helpers.R')

# Convert data to numeric
clean_data <- df_to_numeric(clean_data)


# Convert Churn to factor
clean_data$Churn <- factor(mapvalues(clean_data$Churn, from=c(0, 1), to=c('No', 'Yes')))

# Split data
set.seed(14)
clean_data<- clean_data[ sample(nrow(clean_data)), ]
split <- round(nrow(clean_data) * .75)
train_data <- clean_data[1:split, ]
test_data <- clean_data[(split+1):nrow(clean_data),]

# All models
models <- c('glmnet',    # Logistic regression
            'rpart',     # Decision trees
            'ranger',    # Random forest
            'svmLinear', # Support Vector Machines with Linear Kernel
            'svmRadial', # Support Vector Machines with Radial Basis Function Kernel
            'svmPoly',   # Support Vector Machines with Polynomial Kernel
            'kknn')      # k-Nearest Neighbors

# Fit caret models
results <- fit_caret(models, folds=3, pca=FALSE)


### Neural network

# Transform data to fit neural network
data_x_train <- train_data %>% select(-Churn) %>% as.matrix()
Churn_train <- train_data %>% select(Churn)
data_y_train <- Churn_train %>% select(Churn) %>% onehot() %>% predict(Churn_train)%>% as.matrix()
data_x_test <- test_data %>% select(-Churn)%>% as.matrix()
Churn_test <- test_data %>% select(Churn)
data_y_test <- Churn_test %>% select(Churn) %>% onehot() %>% predict(Churn_test)%>% as.matrix()

# Scale variables
data_x_test <- t(t(t(t(data_x_test) - colMeans(data_x_train))) / apply(data_x_train, 2, sd))
data_x_train <- t(t(t(t(data_x_train) - colMeans(data_x_train))) / apply(data_x_train, 2, sd))

# Hyper-parameters
neurons <- c(128, 64)
learnig_rate <- 0.0025
dropout_keep_prob <- 0.4
epochs <- 10
n_features <- ncol(data_x_train)
n_classes <- ncol(data_y_train)

# Initialize new model
set.seed(14)
model <- init_model()

# train model and get results
model <- train_model(model, verbose=1)
nn_results <- model$nn_results

# Plot neural network results
nn_results %>%
    gather(Metric, Values, -c('Iterations', 'Set')) %>%
    ggplot(aes(x=Iterations, y=Values, color=Set)) + 
    geom_line(size=1.25) + 
    facet_wrap(~Metric) + 
    scale_color_brewer(palette="Set1") + 
    ggthemes::theme_few() +
    labs(title='Neural network performance',
         y='Metric values')


# Predict outputs
pred <- forward_prop(data_x_test, model)$y[,1]
# Predict thresholded values
p <- ifelse(pred < 0.5, 0, 1)
# Confusion matrix
cm <- confusionMatrix(p, data_y_test[,1], positive='1')
# Area under curve
auc <- roc(test_data$Churn, pred)
# Model results
results_neuralnet <- c(cm$overall['Accuracy'], AUC=auc$auc, cm$byClass)
results_all <- rbind(results, results_neuralnet)
results_all[nrow(results_all), ncol(results_all)] <- 'neural_net'

# Models comparison
results_all %>%
    select(Model, Accuracy, Sensitivity, Specificity, AUC) %>%
    gather(Metric, Values, -Model) %>%
    ggplot(aes(x=Model, y=Values)) + 
    geom_bar(stat='identity') + 
    geom_text(aes(x = Model, y=Values, label=round(Values, 3), vjust = -1), size=3) + 
    ylim(0, 1.2) + 
    facet_wrap(~Metric) + 
    ggthemes::theme_few() +
    labs(title='Prediction models comparison',
         x='Prediction models',
         y='Metric values') + 
    theme(axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0.2),
          axis.title.x = element_blank()) +
    scale_x_discrete(labels=c("glmnet" = "Logistic regression", 
                              "rpart" = "Decision trees",
                              "ranger" = "Random forest",
                              'svmLinear' = 'SVM Linear',
                              'svmRadial' = 'SVM Radial',
                              'svmPoly' = 'SVM Polynomial',
                              'kknn' = 'k-Nearest Neighbors',
                              'neural_net' = 'Neural network'))


# SHINY
library(shiny)
# User interface
ui <- shinyUI(pageWithSidebar(
    
    # Application title
    headerPanel("Probability threshold choice"),
    
    # Sidebar 
    sidebarPanel(
        
        # slider input
        sliderInput("obs", 
                    "Probability threshold:", 
                    min = 0,
                    max = 1, 
                    value = 0.5), 
        
        # Metrics table output
        tableOutput("view")
    ),
    
    # Show plots
    mainPanel(
        plotOutput("distPlot1"),
        plotOutput("distPlot2")
    )
))



server <- shinyServer(function(input, output) {
    
    # Plot 1
    output$distPlot1 <- renderPlot({
        
        cm <- confusionMatrix(ifelse(pred < input$obs, 0, 1), data_y_test[,1])
        customes_tbl <- as.data.frame(cm$table)
        customes_tbl[1:2, 'Prediction'] <-  mapvalues(customes_tbl[1:2, 'Prediction'], from=c(0, 1), to=c(1, 0))
        customes_tbl %>%
            ggplot(aes(x=Reference, y=Freq, fill=Prediction)) + 
            geom_bar(stat='identity') +
            ggthemes::theme_few() +
            scale_fill_brewer(palette="Set1",labels=c("Wrong", "Correct")) + 
            labs(title='Prediction threshold evaluation',
                 y='Customers') + 
            theme(axis.title.x = element_blank()) + 
            
            scale_x_discrete(labels=c("0" = "Not churned", 
                                      "1" = "Churned")) 
        
        })
    
    # Plot 2
    output$distPlot2 <- renderPlot({
        
        cm <- confusionMatrix(ifelse(pred < input$obs, 0, 1), data_y_test[,1])
        metrics <- as.data.frame(c(cm$byClass, cm$overall))
        metrics$Metric <- row.names(metrics)
        colnames(metrics) <- c('Value', 'Metric')
        
        metrics %>%
            filter(Metric %in% c('Accuracy', 'Sensitivity', 'Specificity')) %>%
            ggplot(aes(x=Metric, y=Value)) + 
            geom_bar(stat='identity') + 
            ylim(0, 1.1) +
            ylab('Metric value') +
            ggtitle('Model metrics comparison') + 
            ggthemes::theme_few()  + 
            geom_text(aes(x = Metric, y=Value, label=round(Value, 3), vjust = -1), size=5) + 
            theme(axis.title.x=element_blank()) 
        
        })
    
    # Matrics table
    output$view <- renderTable({
        cm <- confusionMatrix(ifelse(pred < input$obs, 0, 1), data_y_test[,1])
        metrics <- as.data.frame(c(cm$byClass, cm$overall))
        metrics$Metric <- row.names(metrics)
        colnames(metrics) <- c('Value', 'Metric')
        metrics[,2:1]
        
    })
})

# Run shiny
shinyApp(ui, server)





