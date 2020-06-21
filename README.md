This is the author's implementation of the PAMELI algorithm proposed in [PAMELI: A Meta-Algorithm for Computationally Expensive Multi-Objective Optimization Problems](https://arxiv.org/abs/1707.06600).

## Usage example:

`cd` to the directory of the repository and run:

`python run.py --problem <PROBLEM_NAME>`

In the current implementation you can use any of the DTLZ and WFG problems. For example to test on DTLZ2:

`python run.py --problem DTLZ1`

## Results
PAMELI vs. KRVEA on the DTLZ problem set: 

![](PAMELIvsKRVEA.png)

Evolution for 10 iterations of the approximated Pareto set on the DTLZ2 problem:

![](dtlz2.gif)
