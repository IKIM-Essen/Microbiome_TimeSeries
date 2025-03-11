# Machine learning prediction on microbial time series

Scripts for creating and analyzing machine learning models for the prediction of microbial time series data.
Input are microbial abundances in a tsv file created from a BIOM file. The environmental model can process metadata as well provided in a separate file.

The intention is to compare different types of machine learning models for their ability to predict microbial abundances as well as changes in microbial community composition. These models could built the basis for an early warning system, usable for human as well as environmental samples, based on outlier detection.

Models have been trained and tested on published time series data from the following references:
Caporaso et al., Genome Biology, 2011
David et al., Genome Biology, 2014
Kodera et al., Environ Microbiome, 2023
LaMartina et al., Microbiome, 2021