#### read the data ####

banana = read.csv('banana.csv')

#### plot 1,2 ####
#### new table for magnitude changes on each day ####

mag_change = matrix(0, ncol = 5, nrow = 12)

mag_change[,1] = banana$X10.20_magnitude - banana$X10.19_magnitude
mag_change[,2] = banana$X10.21_magnitude - banana$X10.20_magnitude
mag_change[,3] = banana$X10.22_magnitude - banana$X10.21_magnitude
mag_change[,4] = banana$X10.23_magnitude - banana$X10.22_magnitude
mag_change[,5] = banana$X10.24_magnitude - banana$X10.23_magnitude

#### treatment information #### 

tr_ctr = c(0,0,3,3,1,2,2,2,3,0,1,1)


#### functions for permutation tests ####
set.seed(2022)

perm.test.mean = function(outcomes, treatments, nrep = 1000){
  # permutation test with difference-in-means
  n = length(outcomes)
  control = which(treatments == 0)
  treatments[setdiff(1:n, control)] = 1
  
  real_val = mean(outcomes[which(treatments == 1)]) - 
    mean(outcomes[which(treatments == 0)])
  all_perm = matrix(0, nrow = nrep, ncol = n)
  for (i in 1:nrep) {
    all_perm[i,] = permute(treatments)
  }
  
  n_perm = dim(all_perm)[1]
  diff_list = rep(0, n_perm)
  
  for (i in 1:n_perm) {
    diff_list[i] = mean(outcomes[which(all_perm[i,] == 1)]) - 
      mean(outcomes[which(all_perm[i,] == 0)])
  }
  
  p.value = length(which( diff_list <= real_val)) / n_perm
  return(p.value)
}


perm.test.median = function(outcomes, treatments, nrep = 1000){
  # permutation test with difference-in-means
  n = length(outcomes)
  control = which(treatments == 0)
  treatments[setdiff(1:n, control)] = 1
  
  real_val = median(outcomes[which(treatments == 1)]) - 
    median(outcomes[which(treatments == 0)])
  all_perm = matrix(0, nrow = nrep, ncol = n)
  for (i in 1:nrep) {
    all_perm[i,] = permute(treatments)
  }
  n_perm = dim(all_perm)[1]
  diff_list = rep(0, n_perm)
  
  for (i in 1:n_perm) {
    diff_list[i] = median(outcomes[which(all_perm[i,] == 1)]) - 
      median(outcomes[which(all_perm[i,] == 0)])
  }
  
  p.value = length(which( diff_list <= real_val)) / n_perm
  return(p.value)
}


## tests for apple vs control ## 

p.value.list1.mean = rep(0,5)
p.value.list1.median = rep(0,5)
tr01 = c(which(tr_ctr == 0), which(tr_ctr == 1))
for (i in 1:5) {
  p.value.list1.mean[i] = perm.test.mean(outcomes = mag_change[tr01, i], 
                                         treatments = tr_ctr[tr01]) 
  p.value.list1.median[i] = perm.test.median(outcomes = mag_change[tr01, i], 
                                         treatments = tr_ctr[tr01]) 
}


## plot(p.value.list1.mean, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in mean test on apple treatment')
## plot(p.value.list1.median, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in median test on apple treatment')

## tests for cucumber vs control ## 

p.value.list2.mean = rep(0,5)
p.value.list2.median = rep(0,5)
tr02 = c(which(tr_ctr == 0), which(tr_ctr == 2))
for (i in 1:5) {
  p.value.list2.mean[i] = perm.test.mean(outcomes = mag_change[tr02, i], 
                                         treatments = tr_ctr[tr02]) 
  p.value.list2.median[i] = perm.test.median(outcomes = mag_change[tr02, i], 
                                             treatments = tr_ctr[tr02]) 
}

## plot(p.value.list2.mean, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in mean test on cucumber treatment')
## plot(p.value.list2.median, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in median test on cucumber treatment')



## tests for mixed vs control ## 

p.value.list3.mean = rep(0,5)
p.value.list3.median = rep(0,5)
tr03 = c(which(tr_ctr == 0), which(tr_ctr == 3))
for (i in 1:5) {
  p.value.list3.mean[i] = perm.test.mean(outcomes = mag_change[tr03, i], 
                                         treatments = tr_ctr[tr03]) 
  p.value.list3.median[i] = perm.test.median(outcomes = mag_change[tr03, i], 
                                             treatments = tr_ctr[tr03]) 
}


## plot(p.value.list3.mean, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in mean test on mixed treatment')
## plot(p.value.list3.median, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in median test on mixed treatment')

dev.new()
pdf("1.pdf")
plot(c(1,2,3,4,5), p.value.list1.mean, type='l', col = 'red', lty = 1, lwd = 3,
     xlab = 'Day', ylab = 'p-value', main = 'Difference-in-means tests for magnitude',
    ylim = c(0, 1), cex.lab = 1.4, cex.axis = 1.2)
lines(c(1,2,3,4,5), p.value.list2.mean, type='l', col="blue", lty = 1, lwd =3)
lines(c(1,2,3,4,5), p.value.list3.mean, type='l', col="black", lty = 1, lwd =3)
legend(x=1.1, y = 0.1, legend = c('apple vs control'), lty = 1, col = c('red'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.17, legend = c('cucumber vs control'), lty = 1, col = c('blue'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.24, legend = c('mixed vs control'), lty = 1, col = c('black'),  bty="n",lwd =3,cex= 1.3)
dev.off()

dev.new()
pdf("2.pdf")
plot(c(1,2,3,4,5), p.value.list1.median, type='l', col = 'red', lty = 1, lwd = 3,
     xlab = 'Day', ylab = 'p-value', main = 'Difference-in-medians tests for magnitude',
     ylim = c(0, 1), cex.lab = 1.4, cex.axis = 1.2)
lines(c(1,2,3,4,5), p.value.list2.median, type='l', col="blue", lty = 1, lwd =3)
lines(c(1,2,3,4,5), p.value.list3.median, type='l', col="black", lty = 1, lwd =3)
legend(x=1.1, y = 0.1, legend = c('apple vs control'), lty = 1, col = c('red'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.17, legend = c('cucumber vs control'), lty = 1, col = c('blue'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.24, legend = c('mixed vs control'), lty = 1, col = c('black'),  bty="n",lwd =3,cex= 1.3)
dev.off()



#### plot 3,4 ####
#### new table for magnitude changes on each day ####

mag_change = matrix(0, ncol = 5, nrow = 12)

mag_change[,1] = banana$X10.20_g - banana$X10.19_g
mag_change[,2] = banana$X10.21_g - banana$X10.20_g
mag_change[,3] = banana$X10.22_g - banana$X10.21_g
mag_change[,4] = banana$X10.23_g - banana$X10.22_g
mag_change[,5] = banana$X10.24_g - banana$X10.23_g

#### treatment information #### 

tr_ctr = c(0,0,3,3,1,2,2,2,3,0,1,1)


#### functions for permutation tests ####
set.seed(2022)


## tests for apple vs control ## 

p.value.list1.mean = rep(0,5)
p.value.list1.median = rep(0,5)
tr01 = c(which(tr_ctr == 0), which(tr_ctr == 1))
for (i in 1:5) {
  p.value.list1.mean[i] = perm.test.mean(outcomes = mag_change[tr01, i], 
                                         treatments = tr_ctr[tr01]) 
  p.value.list1.median[i] = perm.test.median(outcomes = mag_change[tr01, i], 
                                             treatments = tr_ctr[tr01]) 
}


## plot(p.value.list1.mean, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in mean test on apple treatment')
## plot(p.value.list1.median, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in median test on apple treatment')

## tests for cucumber vs control ## 

p.value.list2.mean = rep(0,5)
p.value.list2.median = rep(0,5)
tr02 = c(which(tr_ctr == 0), which(tr_ctr == 2))
for (i in 1:5) {
  p.value.list2.mean[i] = perm.test.mean(outcomes = mag_change[tr02, i], 
                                         treatments = tr_ctr[tr02]) 
  p.value.list2.median[i] = perm.test.median(outcomes = mag_change[tr02, i], 
                                             treatments = tr_ctr[tr02]) 
}

## plot(p.value.list2.mean, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in mean test on cucumber treatment')
## plot(p.value.list2.median, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in median test on cucumber treatment')



## tests for mixed vs control ## 

p.value.list3.mean = rep(0,5)
p.value.list3.median = rep(0,5)
tr03 = c(which(tr_ctr == 0), which(tr_ctr == 3))
for (i in 1:5) {
  p.value.list3.mean[i] = perm.test.mean(outcomes = mag_change[tr03, i], 
                                         treatments = tr_ctr[tr03]) 
  p.value.list3.median[i] = perm.test.median(outcomes = mag_change[tr03, i], 
                                             treatments = tr_ctr[tr03]) 
}


## plot(p.value.list3.mean, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in mean test on mixed treatment')
## plot(p.value.list3.median, type = 'l', xlab = 'Day', ylab = 'p values', main = 'Difference in median test on mixed treatment')

dev.new()
pdf("3.pdf")
plot(c(1,2,3,4,5), p.value.list1.mean, type='l', col = 'red', lty = 1, lwd = 3,
     xlab = 'Day', ylab = 'p-value', main = 'Difference-in-means tests for g value',
     ylim = c(0, 1), cex.lab = 1.4, cex.axis = 1.2)
lines(c(1,2,3,4,5), p.value.list2.mean, type='l', col="blue", lty = 1, lwd =3)
lines(c(1,2,3,4,5), p.value.list3.mean, type='l', col="black", lty = 1, lwd =3)
legend(x=1.1, y = 0.1, legend = c('apple vs control'), lty = 1, col = c('red'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.17, legend = c('cucumber vs control'), lty = 1, col = c('blue'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.24, legend = c('mixed vs control'), lty = 1, col = c('black'),  bty="n",lwd =3,cex= 1.3)
dev.off()

dev.new()
pdf("4.pdf")
plot(c(1,2,3,4,5), p.value.list1.median, type='l', col = 'red', lty = 1, lwd = 3,
     xlab = 'Day', ylab = 'p-value', main = 'Difference-in-medians tests for g value',
     ylim = c(0, 1), cex.lab = 1.4, cex.axis = 1.2)
lines(c(1,2,3,4,5), p.value.list2.median, type='l', col="blue", lty = 1, lwd =3)
lines(c(1,2,3,4,5), p.value.list3.median, type='l', col="black", lty = 1, lwd =3)
legend(x=1.1, y = 0.1, legend = c('apple vs control'), lty = 1, col = c('red'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.17, legend = c('cucumber vs control'), lty = 1, col = c('blue'),  bty="n",lwd =3,cex= 1.3)
legend(x=1.1, y = 0.24, legend = c('mixed vs control'), lty = 1, col = c('black'),  bty="n",lwd =3,cex= 1.3)
dev.off()

