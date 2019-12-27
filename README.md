
# GoSelect
![Build status](https://travis-ci.org/davidkleiven/goselect.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/davidkleiven/goselect/badge.svg?branch=master)](https://coveralls.io/github/davidkleiven/goselect?branch=master)

Simple Go library for model selection. **GoSelect** implements the following selection algorithms

* Branch and Bound using the modified Afaike's Information Criterion (AICC) as cost function
* Simmulated Annealing using AICC as the cost function
* LASSO (both LARS and coordinate descent)

# Data Format
Many of the command line tools implemented in **GoSelect** reads data from a comma 
separated text files. It is assumed that the file has a header that includes a name
for each feature. An example of a valid CSV file is shown below

```
# Age, Height, Weight
   20,    184,     80
   40,    192,     92
   60,    188,     88
```

# Command Line Tools
The following command line tools are available in **GoSelect**

* **goselect-bnb** performs model selection using a branch and bound method
* **goselect-cohenlasso** calculates Cohen's kappa using the lasso method (for the model by minimising AICC)
* **goselect-lasso** runs the LASSO alggorithms
* **goselect-mem** memory estimate for the queue used in branch and bound method
* **goselect-nestelasso** runs nested lasso (e.g. sequential LASSO by exlcluding the least relevant features after each run)
* **goselect-plotlasso** plots results from LASSO  runs
* **goselect-sa** runs model selection using simmulated annealing ny minimizing AICC

A detailed information of the arguments to each CLI tool can be found by running

```
goselect-bnb -h
```
