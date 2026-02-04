
# data generating function
genY <- function(x) {
    1.2 * sin(x[,1]) + x[,2]^2 + exp(-x[,3]) + 
    x[,4] * x[,5] + 1.5*log(abs(x[,6])+1) + 
    2*x[,8] + x[,5] * x[,9]-1.75*x[,9] + 1.5*pmin(1, x[,7])
}


set.seed(123)
# sample 1000 values for 15 variables
features <- matrix(rnorm(15000), ncol=15, dimnames=list(NULL, paste0("x", 1:15)))
# compute the true response to these features 
target <- genY(features)
# add error
error <- rnorm(1000, sd=0.25^2)
target <- target + error

# combine target and predictors in a data.frame
d <- data.frame(target=target, features)

# split data in training (700 observations) and testing (300 observations) 
i <- sample(1000)[1:700]
train <- d[i, ]
test  <- d[-i, ]

# fit a random forest model without tuning
rf <- randomForest::randomForest(target~., data=train,  importance=TRUE)


##### function to compute permutation importance of variables
perm_imp <- function(model, data, y, n=10) {
  # baseline mse
	mse <- mean((predict(model, data) - y)^2)
	# loop over all predictor variables
	perm <- lapply(names(data), \(v) {
	  # make a copy of the data
		dv <- data 
		# replicate n times
		r <- replicate(n, {
		  # randomize the values of variable v
			dv[v] <- sample(dv[[v]])
			# make prediction 
			p <- predict(model, dv)
			# compute mse
			mean((p - y)^2)
		})
	})
	# combine output for all variables
	out <- do.call(cbind, perm)
	# compute the % change in mse
	out <- 100 * (out-mse) / mse
	colnames(out) <- names(data)
	out
}

imp <- perm_imp(rf, test[,-1], test$target, n=25)
# get the mean MSE change for each predictor
imp_mean <- colMeans(imp)
# get the range and sort them in order of importance
imp_range <- apply(imp, 2, range)[, order(imp_mean)]

# plot
barplot(sort(imp_mean), horiz=T, las=1, col="blue")

# built-in RF plot
# Also note the purity estimate (here the decrease in variance), 
# in ISLR they only talk about purity in the context of classification
imp_rf <- randomForest::varImpPlot(rf)

# note the differences with my estimate. 
# can anyone explain that?
plot(imp_rf[,1], imp_mean)


plot(sort(imp_mean), 1:length(imp_mean), axes=FALSE, xlab="Decrease in RMSE (%)", 
     ylab="", xlim=range(imp_range), cex=1.5, pch=20, col="red")
axis(1); axis(2, at = 1:length(imp_mean), labels=names(imp_mean), las=1)
s <- sapply(1:ncol(imp_range), \(i) lines(cbind(imp_range[,i], i), lwd=1.5))

##########
##
## partial response plots
##

pdp <- function(model, data, steps) {	
  # for each variable
	pd <- lapply(names(data), \(v) {
	  # get range of values to make predictions for
	  r <- range(data[[v]])
	  # extent the range a little
	  r <- r + c(-0.1, 0.1) * diff(r)
	  # get the intermediate values
		x <- seq(r[1], r[2], length.out=steps)
		
		dv <- data 
		# for each value in x
		z <- data.frame(x=x, 
			sapply(x, \(i) { 
			  #set the value of v to x[i]
				dv[[v]] <- i
				# get the average prediction 
				mean(predict(model, dv))
			})
		)
		colnames(z) <- c(v, "target")
		z
	})
	names(pd) <- names(data) 
	pd
}

pd <- pdp(rf, test[,-1], 25)

# each plot its own y-range
par(mfrow=c(4,4), mar=c(3,3,1,1), las=1, mgp = c(2, 1, 0), cex.axis=.9)
s <- sapply(pd, \(x) plot(x, type="l", col="red", lwd=2))

# each plot the same y-range (much better!)
par(mfrow=c(4,4), mar=c(3,3,1,1), las=1, mgp = c(2, 1, 0), cex.axis=.9)
r <- range(sapply(pd, \(x) range(x[,2])))
s <- sapply(pd, \(x) plot(x, type="l", ylim=r, col="red", lwd=2))


# built-in randomForest method
x11() # show in separate window to compare with above
par(mfrow=c(4,4), mar=c(4,4,1,1), las=1, mgp = c(2, 1, 0), cex.axis=.9)
s <- for (i in 1:length(n)) {
  randomForest::partialPlot(rf, test, n[i], main="",
                      xlab=n[i], ylim=r, col="red")
}

###
### Individual Conditional Expectation (ICE)
###

ice <- function(model, data, steps=50) {	
	# for each variable
	pd <- lapply(names(data), \(v) {
	  # get a sequence of x values
		x <- seq(min(data[[v]]), max(data[[v]]), length.out=steps)
		dv <- data 
		# for each x value predict
		z <- lapply(x, \(i) { 
			dv[[v]] <- i
			predict(model, dv)
		})
		# combine
		z <- data.frame(step=x, do.call(rbind, z))
		colnames(z)[1] <- v
		z
	})
	names(pd) <- names(data)
	pd
}

ic <- ice(rf, test[,-1], 10)
i <- 2
blue <- rgb(0, 0, .5, alpha=.25)
plot(ic[[i]][,1], ic[[i]][,2], ylim=range(ic[[i]][,-1]), col=blue, type="l", xlab=names(ic[[i]][1]), ylab="Partial dependence")
s <- sapply(3:ncol(ic[[i]]), \(j) lines(ic[[i]][,1], ic[[i]][,j], col=blue))
lines(ic[[i]][,1], rowMeans(ic[[i]][,-1]), col="red", lwd=3)

 
### SHAP
library(kernelshap)
library(shapviz)

shap <- permshap(rf, X=test[,-1], bg_X = test[sample(nrow(test), 50),-1])
sv <- shapviz(shap) # this takes a while

#three plots
sv_importance(sv, kind = "bee")
sv_importance(sv, kind = "bar")
sv_dependence(sv, v = "x2") 


