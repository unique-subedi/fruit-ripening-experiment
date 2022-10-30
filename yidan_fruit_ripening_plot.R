library(ggplot2)
library(tibble)
library(tidyr)

df = read.csv("g_df_melted.csv")
df = df[13:72,]
df$number = as.factor(df$number)
df$treatment = as.factor(df$treatment)
df$covariates = as.numeric(as.factor(df$covariates))

ggplot(df, aes(x=covariates, y=value, color=number)) +
    # facet_wrap(~treatment, ncol=4) +
    geom_point(position=position_dodge(1), alpha=0.7) + 
    # geom_smooth(se=FALSE, method=loess) +
    geom_line() +
    labs(x = "Time point", y = "Lag 1 Difference of Measurements") +
    ggtitle("Normalised Green RGB Value") +
    theme_classic() 
