
## The plan
0. Baselines:
    1. MUST evaluate retrieval on baseline models!
    2. Generate tables
    3. 
    
1.PCA
    1. Create parallel code that:
        1. Uses the backbones of the baselines
        2. And then evaluate on retrieval directly
        3. Generate tables for R@K and mAP
2. KD
    1. Train the pipeline
        1. Do it for all sizes if you can (?) 
        2. Then one by one, evaluate on retrieval
        3. Generate graphs and tables
3. AC:
    1. Expirement with an autoencoder pipeline
    2. Woud a very lightweight masked autoencoder pipeline also work here?
        1. This could go into future work section


### Details:
# PCA
1. Copy the main script
2. Remove items unnecessary to this new pipeline
3. Generate pca objects for sizes from 2048 to 8
4.
4. Evaluate

NYD (Where I stopped):
1. There's definitely a problem in calculating the metrics of retrieval
2. I've tried both using torchemetrics and using functions from scratch. They both give identical results that are wrong
3. PCA experiments gives better results for mAP if in DEBUG MODE (small data), and very inaccurate results when using the full data
4. Start working on the KD pipeline