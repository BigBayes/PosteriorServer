# Posterior Server and SNEP

This github repository contains code for stochastic natural gradient expectation propagation, a novel algorithm for distributed Bayesian learning (see http://arxiv.org/abs/1512.09327). In addition, there is code for various SGD methods such as asynchronous SGD (https://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf) and elastic averaging SGD (https://papers.nips.cc/paper/5761-deep-learning-with-elastic-averaging-sgd.pdf). The code is research code that is fairly modular to allow testing on various different models. It is not production code meant to compete eg with standard neural network packages. We believe it should be possible to implement our algorithm much more efficiently in such a package.

The experiments folder contains experiments script to reproduce the experiments in the paper (to come).

## First steps

The code uses julia v0.4.7 (http://julialang.org ). Clone the repo and run `install.jl` to install all the necessary julia packages.

Unzip the data files in /data and set the data_path string in scripts/path.jl to this or another convenient directory

To run a simple test experiment open a terminal, change to the scripts directory and run

```julia run.jl mnist_100_test```

The results will be displayed in the terminal while the experiment is running and will be saved to the results folder at the end of the experiment.

## Reproducing experiments

Run the experiments as shown above (eg `julia run.jl mnist/#workers/mnist_500x300_base`). There are plotting scripts in the scripts dir that should automatically reproduce the figures in the paper. **Note that reproducing the results will take a long time and considerable computational resources.** 

## Code structure

The code is modular. The main algorithm is implemented in `PosteriorServer.jl` and various smaller files. It interacts with the model and the data through a data model which combines both. Each data model has to implement a sampling method. A number of standard samplers can be found in `SGMCMC.jl`. A data model also has to implement some evaluation methods for a test dataset. This enables us to track test performance throughout the training process.

To start an experiment, run `julia run.jl <experiment-name>`. `run.jl` set up the right paths (by calling `paths.jl`) and then iterates over the values of a hyperparameter under investigation in the current experiment. For each parameter value, it starts an experiment by calling the appropriate run script (eg `runMochaPosteriorServer.jl` for the neural network experiments). This run script will set up the split up the data, set up the different processes for the workers and the corresponding data models and eventually run `PosteriorServer.run()`. 

The synchronisations are implemented using an intermediate sync facility which receives requests for a synchronisation and performs the synchronisation. The worker process will request synchronisations which will be executed if the posterior server is not busy. 

A similar facility handles request for evaluation of train/test performance.

Experimental results are saved in the results folder and can be loaded using the `read_*.jl` scripts in the scripts folder.

For more details, please have a look at the comments in the code or send us an email.
