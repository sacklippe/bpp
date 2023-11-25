# The Bin Packing Problem (BPP)

In the [BPP](https://en.wikipedia.org/wiki/Bin_packing_problem), a collection of items must be stored in the minimum possible number of bins. In this case, the items have a weight associated and the bins are restricted to carry a maximum weight. This problem has many real-world applications such as loading trucks with a weight restriction, container scheduling, or design of FPGA chips. In terms of complexity, the BPP is an NP-hard problem and its formulation is given by

$$
\min \sum_{j=1}^m y_j,\tag{1}
$$

subject to:

$$
\sum_{i=1}^n w_i x_{ij} \le W y_j \qquad  \forall j=1,...,m,\tag{2}
$$

$$
\sum_{j=1}^m x_{ij} = 1  \qquad \forall i = 1, ..., n,\tag{3}
$$

$$
x_{ij}\in  \{0,1\} \qquad \forall i=1,..,n \qquad \forall j=1,..,m,\tag{4}
$$

$$
y_{j}\in  \{0,1\} \qquad \forall j=1,..,m, \tag{5}
$$

where $n$ is the number of items (nodes), $m$ is the number of bins, $w_{i}$ is the i-th item weight, $W$ is the bin capacity, $x_{ij}$ and $y_j$ are binary variables that represent if the item $i$ is in the bin $j$, and whether bin $j$ is used or not, respectively. From the above equations, Eq.(10) is the cost function to minimize the number of bins, Eq.(11) is the inequality constraint for the maximum weight of a bin, Eq.(12) is the equality constraint to restrict that an item is only in one of the bins, and Eqs.(13) and (14) means that $y_i$ and $x_{ij}$ are binary variables.
