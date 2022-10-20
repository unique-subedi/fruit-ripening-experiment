## Tests we can conduct

1. Test wether there is a within-subject difference in *time* for each treatment.
    - paired sample, before/after treatment is applied 
    - test separately for each treatment (randomise within each pair on before/after)
    - in this case, we have 3 paired samples for each treatment
    - how to meaningfully compare p-values across treatments if we are only randomising time? In this case treatment is being controlled for?
    - use mxied effect model, or s.e. adjusted for within treatment grouped effect for t-test statistics.
2. Test wether there is a between-subject difference in *pairwise treatment assignment* across time
    - grouped sample for each time point
        - control: banana by itself, treatment: banana with apple;
        - control: banana by itself, treatment: banana with cucumber;
        - control: banana by itself, treatment: banana with cucumber+apple.
    - test mean of difference for every two paired control and treatment (randomise within each time point of treatment assignment)
        - i.e. $H_0: \tau_1=0, \tau_2=0, \tau_3=0$
    - adjust for multiplicity of the three tests
    - how to deal with p-value generated across time?
        - we can only meaningfully compare p-values at each time points, not across time.
        - we could possibly alter the null hypothesis to also include each time point comparison??
3. between-subject difference in *treatment assignment* across time
    - grouped sample for each time point
    - randomise treatment assignment for *all* treatments. What is an appropriate test statistics?
4. test for interaction between time and treatment
    - F1-LD-F1 design as in this [R package](https://www.jstatsoft.org/article/view/v050i12).

