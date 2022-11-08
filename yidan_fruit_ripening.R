library(ggplot2)
library(tibble)
library(tidyr)

g_df_melted = read.csv("g_df_melted.csv")
mag_df_melted = read.csv("mag_df_melted.csv")
g_df_melted$number = as.factor(g_df$number)
mag_df_melted$number = as.factor(mag_df$number)
g_df_melted$covariates = as.numeric(as.factor(g_df$covariates))
mag_df_melted$covariates = as.numeric(as.factor(mag_df$covariates))

ggplot(g_df_melted, aes(x=covariates, y=value, color=number)) +
    facet_wrap(~factor(treatment), ncol=4) +
    geom_point(alpha=0.7) + 
    # geom_smooth(se=FALSE, method=loess) +
    geom_line() +
    labs(x = "Time point", y = "Lag 1 Difference of Measurements") +
    ggtitle("Green RGB Value") +
    theme_classic() 

ggplot(mag_df_melted, aes(x=covariates, y=value, color=number)) +
    facet_wrap(~treatment, ncol=4) +
    geom_point(alpha=0.7) + 
    geom_line() +
    labs(x = "Time point", y = "Lag 1 Difference of Measurements") +
    ggtitle("RGB Magnitude Value") +
    theme_classic() 

ggplot(g_df_melted) +
    geom_histogram(aes(x=value, color=factor(covariates), fill=factor(covariates)), alpha=0.5, position="dodge") +
    labs(x = "Lag 1 Difference of Green RGB") +
    ggtitle("Histogram for each Day of Measurements") +
    labs(colour = "Day", fill="Day") +
    theme_classic() 

ggplot(mag_df_melted) +
    geom_histogram(aes(x=value, color=factor(covariates), fill=factor(covariates)), alpha=0.5, position="dodge") +
    labs(x = "Lag 1 Difference of RGB Magnitude") +
    ggtitle("Histogram for each Day of Measurements") +
    labs(colour = "Day", fill="Day") +
    theme_classic() 


