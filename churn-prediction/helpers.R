# Helper functions for churn prediction


#   TABLE OF CONTENT:
#   - Function to convert factors to numeric.
#   - Fuction to fit different caret models.
#   - Neural nnetwork functions.




# Covert data to numeric - (factors to one-hot)
df_to_numeric <- function(df) {
    
    for (i in names(df)) {
        
        # if numeric
        if (class(df[,i]) != 'factor')
            df[,i] <- as.numeric(df[,i])
        
        # if factor with 2 levels
        else if (class(df[,i]) == 'factor' & length(levels(df[,i])) == 2) {
            suppressMessages(
                val <- mapvalues(df[,i], from=c('No','Male', 'Yes', 'Female'), to=c(0, 0, 1, 1)))
            val <- as.numeric(as.character(val))
            df[,i] <- val }
        
        # if factor with more than 2 levels    
        else if (class(df[,i]) == 'factor' & length(levels(df[,i])) > 2) {
            val <- df[,i]
            lvl <- levels(df[,i])
            lvl <- lvl[lvl != "No"]
            df[,i] <- NULL
            
            for (j in lvl) {
                df <- cbind(df, as.numeric(val == j))
                names(df)[ncol(df)] <- paste0(i, '_', j) }
        }} 
    
    # remove wrong characters from variable names
    names(df) <- gsub('-', '_', names(df))
    names(df) <- gsub(' ', '_', names(df))
    names(df) <- gsub('\\(', '', names(df))
    names(df) <- gsub('\\)', '', names(df))
    return(df)}


# Fit different caret:: models and get results
fit_caret <- function(models, folds=5, pca=FALSE) {
    
    results <- c()
    for (m in models) {
        
        set.seed(14)
        
        preProcess = c('center', 'scale')
        # check pca argument
        if (pca) preProcess <- c(preProcess, 'pca')
        
        # fit model
        model <- train(
            Churn ~ ., data = train_data,
            method = m,
            metric = 'ROC',
            trControl = trainControl(
                method = "cv", number = folds,
                summaryFunction = twoClassSummary,
                classProbs = T,
                verboseIter = TRUE),
            preProcess = preProcess)
        
        
        # predict thresholded values
        p <- predict.train(model, test_data, type='raw')
        # predict probability values
        p_prob <- predict.train(model, test_data, type='prob')[,1]
        # confusion matrix
        cm <- confusionMatrix(p, test_data$Churn)
        # area under curve
        auc <- roc(test_data$Churn, p_prob)
        # model results
        results_model <- c(cm$overall['Accuracy'], AUC=auc$auc, cm$byClass)
        # append results to mian table
        results <- rbind(results, results_model)
    }
    
    # convert results to data frame
    row.names(results) <- 1:length(models)
    results <- as.data.frame(results)
    results$Model <- models
    
    return(results)
}






### Naural network functions

# ReLu activation
relu <- function(x) x * (x > 0)

# ReLu activation derivative
relu_der <- function(x) 1 * (x > 0)

# Softmax function
softmax <- function(x) {
    exp_sum <- rep(rowSums(exp(x)), ncol(x))
    dim(exp_sum) <- dim(x)
    return(exp(x)/exp_sum) }

# Cost function - mean squared error
cost_mse <- function(labels, predictions) 
    return(mean((labels - predictions)^2))

# Dropout
dropout<- function(x, keep_prob) {
    mask <- runif(nrow(x) * ncol(x))
    dim(mask) <- dim(x)
    mask <- mask < keep_prob
    return(x * mask / keep_prob) }

# New model initialization
init_model <- function() {
    model <- list(iters=length(neurons) + 1)
    model$fit_iter <- 1
    sizes <- c(n_features, neurons, n_classes)
    model$sizes <- sizes
    for (i in 1:model$iters) {
        model$w[[i]] <- matrix(data=runif(sizes[i]*sizes[i+1], min=-0.1, max=0.1), nrow=sizes[i])
        model$b[[i]] <- runif(sizes[i+1], min=0, max=0.1)}
    return(model) }

# Forward propagation
forward_prop <- function(x, model, keep_prob=1.0) {
    model$x <- x
    for (i in 1:model$iters) {
        
        if (i==1) model$z[[i]] <- model$x %*% model$w[[i]] + model$b[[i]]
        else model$z[[i]] <- model$a[[i-1]] %*% model$w[[i]] + model$b[[i]]
        
        if (i!=model$iters) model$a[[i]] <- dropout(relu(model$z[[i]]), keep_prob) 
        else model$a[[i]] <- softmax(model$z[[i]]) }
    model$y <- model$a[[i]]
    
    return(model) }

# Back propagation
back_prop <- function(y, model) {
    
    for (i in model$iters:1) {
        
        if (i==model$iters) model$dz[[i]] <- model$a[[i]] - y
        else model$dz[[i]] <- model$dz[[i+1]] %*% t(model$w[[i+1]]) * relu_der(model$z[[i]])
        
        if (i==1) model$dw[[i]] <- t(model$x) %*% model$dz[[i]] / nrow(y)
        else model$dw[[i]] <- t(model$a[[i-1]]) %*% model$dz[[i]] / nrow(y)
        
        model$db[[i]] <- colSums(model$dz[[i]]) / nrow(y) }
    
    return(model) }

# Adam optimizer
Adam_optimizer <- function(model, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.01) {
    
    for (i in 1:model$iters) {
        
        if (model$fit_iter == 1) {
            model$dw_v[[i]] <- model$dw[[i]] * 0
            model$db_v[[i]] <- model$db[[i]] * 0
            model$dw_s[[i]] <- model$dw[[i]] * 0
            model$db_s[[i]] <- model$db[[i]] * 0 }
        
        else {
            model$dw_v[[i]] <- beta1 * model$dw_v[[i]] + (1 - beta1) * model$dw[[i]]
            model$db_v[[i]] <- beta1 * model$db_v[[i]] + (1 - beta1) * model$db[[i]]
            model$dw_s[[i]] <- beta2 * model$dw_s[[i]] + (1 - beta2) * model$dw[[i]] ^ 2
            model$db_s[[i]] <- beta2 * model$db_s[[i]] + (1 - beta2) * model$db[[i]] ^ 2 }
        
        model$dw_vc[[i]] <- model$dw_v[[i]] / (1 - beta1 ^ 2)
        model$db_vc[[i]] <- model$db_v[[i]] / (1 - beta1 ^ 2)    
        model$dw_sc[[i]] <- model$dw_s[[i]] / (1 - beta2 ^ 2)
        model$db_sc[[i]] <- model$db_s[[i]] / (1 - beta2 ^ 2)
        
        model$w[[i]] <- model$w[[i]] - lr * (model$dw_vc[[i]] / (sqrt(model$dw_sc[[i]]) + epsilon))
        model$b[[i]] <- model$b[[i]] - lr * (model$db_vc[[i]] / (sqrt(model$db_sc[[i]]) + epsilon)) }
    model$fit_iter <- model$fit_iter + 1
    return(model) }

# Fit model - one iteration
fit_model <- function(x, y, model, lr, keep_prob) {
    model <- forward_prop(x, model, keep_prob)
    model <- back_prop(y, model)
    model <- Adam_optimizer(model, lr=lr)
    return(model) }        

# Save iteration results
get_results <- function(x, y, model) {
    model <- forward_prop(x, model)
    cost <- cost_mse(model$y, y)
    
    # confusion matrix
    cm <- confusionMatrix(round(model$y[,1]), y[,1], positive='1')
    # Area under curve
    auc <- roc( y[,1], model$y[,1])
    
    accuracy <- mean(ramify::argmax(model$y) == ramify::argmax(y))
    return(list(cost=cost, accuracy=accuracy, sensitivity=cm$byClass['Sensitivity'], 
                specificity=cm$byClass['Specificity'], F1=cm$byClass['F1'], auc=auc$auc)) }  

# Full model training
train_model <- function(model, verbose=0) {
    results_train_all <- c()
    results_test_all <- c()
    
    for (i in 1:epochs) {
        
        # fit one iteratio n
        model <- fit_model(data_x_train, data_y_train, model, learnig_rate, dropout_keep_prob) 
        # get results
        results_train <- get_results(data_x_train, data_y_train, model)
        results_test <- get_results(data_x_test, data_y_test, model)
        if (verbose > 0) {
            print(paste('Epoch:', i, 'train_cost:', results_train$cost, 'train_acc:', results_train$accuracy,
                        'test_cost:', results_test$cost, 'test_acc:', results_test$accuracy)) }
        
        # track train set results
        results_train_all <- rbind(results_train_all, c(Cost=results_train$cost,
                                                        Accuracy=results_train$accuracy,
                                                        results_train$sensitivity,
                                                        results_train$specificity,
                                                        results_train$F1,
                                                        AUC=results_train$auc,
                                                        Iterations=i))
        # track test set results
        results_test_all <- rbind(results_test_all, c(Cost=results_test$cost,
                                                      Accuracy=results_test$accuracy,
                                                      results_test$sensitivity,
                                                      results_test$specificity,
                                                      results_test$F1,
                                                      AUC=results_test$auc,
                                                      Iterations=i)) }
    
    # convert results to appropriate farmat
    results_train_all <- as.data.frame(results_train_all)
    results_test_all <- as.data.frame(results_test_all)
    results_train_all$Set <- 'train'
    results_test_all$Set <- 'test'
    
    nn_results <- union_all(results_train_all, results_test_all)
    
    model$nn_results <- nn_results
    
    return(model)
}






