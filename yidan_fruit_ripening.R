library(ggplot2)
library(tibble)
library(tidyr)

df = read.csv("df_melted.csv")
df$number = as.factor(df$number)
df$treatment = as.factor(df$treatment)
df$covariates = as.numeric(as.factor(df$covariates))

grn_df = df[13:72,]

g = ggplot(grn_df, aes(x=covariates, y=value, color=number)) +
    facet_wrap(~treatment, ncol=4) +
    geom_point(alpha=0.7) + 
    # geom_smooth(se=FALSE, method=loess) +
    geom_line() +
    labs(x = "Time point", y = "Lag 1 Difference of Measurements") +
    ggtitle("Normalised Green RGB Value") +
    theme_classic() 

hist_g = ggplot(grn_df) +
    # facet_wrap(~covariates, ncol=5) +
    geom_histogram(aes(x=value, color=factor(covariates), fill=factor(covariates)), alpha=0.5, position="dodge") +
    labs(x = "Lag 1 Difference of Normalised Green RGB") +
    ggtitle("Histogram for each Day of Measurements") +
    labs(colour = "Day", fill="Day") +
    theme_classic() 

soft_df = df[1:12,]
soft_g = ggplot(soft_df, aes(x=treatment, y=value, color=number, fill=number, size=-value)) +
    # facet_wrap(~treatment, ncol=4) +
    # geom_violin(alpha=0.7) + 
    geom_point(position=position_dodge(0.5), alpha=0.7) +
    # geom_smooth(se=FALSE, method=loess) +
    # geom_line() +
    labs(x = "Treatment", y = "Difference of Softness") +
    ggtitle("Softness") +
    theme_classic() 

ggsave("./figs/green.png", plot=g)
ggsave("./figs/soft.png", plot=soft_g)
ggsave("./figs/hist.png", plot=hist_g)


