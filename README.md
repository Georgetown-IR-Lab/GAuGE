# GAuGE
Genetic Approach using Grounded Evolution

This repository contains the code to reproduce results in the paper 'Genetic Approach to Mitigate Hallucination in Generative IR' accepted at The Second Workshop on Generative Information Retrieval at SIGIR 2024 (Gen-IR@SIGIR24).

The paper can be found here: https://openreview.net/pdf?id=l8P2uJtJRD

## Citation

If you use this code please cite the following paper:

```
@inproceedings{
kulkarni2024genetic,
title={Genetic Approach to Mitigate Hallucination in Generative {IR}},
author={Hrishikesh Kulkarni and Nazli Goharian and Ophir Frieder and Sean MacAvaney},
booktitle={The Second Workshop on Generative Information Retrieval},
year={2024},
url={https://openreview.net/forum?id=l8P2uJtJRD}
}
```

## Run
Set the following parameters in the code:

```
ROUGETYPE: rouge1 or rouge2 ngram overlap metric in the fitness function
DEPTH: Termination Depth
DOC_DEPTH: Sampling Depth
no_of_mutations_per_iteration: mutation budget for each iteration
```

Run the GPT3 and GPT4 file above to obtain results using the respective LLM.

## Data
MS Marco TREC DL 2019, TREC DL 2020 and Dev (small) datasets are used.

## Requirements
Python 3

OpenAI API Key
