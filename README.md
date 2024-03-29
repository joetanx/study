## 1. Search

#### Tree vs Graph Search

- Ref: https://www.cs.rhodes.edu/~kirlinp/courses/ai/s17/handouts/search-algs-uninformed.pdf
- Tree search can be used if the state space is a tree, otherwise graph search must be used.
- All search algorithms (BFS, DFS, uniform-cost, A*, etc) are variations of one of these (usually graph search).
- The only difference between tree search and graph search is that tree search does not need to store the **explored set**, because we are guaranteed never to attempt to visit the same state twice.

#### Measuring problem-solving performance

- **Completeness:** Is the algorithm guaranteed to find a solution when there is one, and to correctly report failure when there is not?
- **Cost optimality:** Does it find a solution with the lowest path cost of all solutions?
- **Time complexity:** How long does it take to find a solution? This can be measured in seconds, or more abstractly by the number of states and actions considered.
- **Space complexity:** How much memory is needed to perform the search?

#### Romania example from AIMA

```mermaid
graph LR

Arad -->|75| Zerind
Zerind -->|71| Oradea
Oradea -->|151| Sibiu
Arad -->|140| Sibiu
Sibiu -->|80| RimnicuVilcea
RimnicuVilcea -->|97| Pitesti
RimnicuVilcea -->|146| Craiova
Craiova -->|138| Pitesti
Arad -->|118| Timisoara
Timisoara -->|111| Lugoj
Lugoj -->|70| Mehadia
Mehadia -->|75| Drobeta
Drobeta -->|120| Craiova
Sibiu -->|99| Fagaras
Fagaras -->|211| Bucharest
Pitesti -->|101| Bucharest
Bucharest -->|90| Giurgiu
Bucharest -->|85| Urziceni
Urziceni -->|142| Vaslui
Vaslui -->|92| Iasi
Iasi -->|87| Neamt
Urziceni -->|98| Hirsova
Hirsova -->|86| Eforie
```

### 1.1. Uninformed Search

|Criterion|Breath-First|Uniform-Cost|Depth-First|Depth-Limited|Iterative-Deepening|Bidirectional (if applicable)|
|---|---|---|---|---|---|---|
|Complete?|Yes|Yes|No|No|Yes|Yes|
|Optimal cost?|Yes|Yes|No|No|Yes|Yes|
|Time|$O(b^d)$|$O(b^{1+[C^*/ϵ]})$|$O(b^m)$|$O(b^ℓ)$|$O(b^d)$|$O(b^{d/2})$|
|Space|$O(b^d)$|$O(b^{1+[C^*/ϵ]})$|$O(bm)$|$O(bℓ)$|$O(bd)$|$O(b^{d/2})$|

### 1.2. Informed Search

|Algorithm|f(n)|Weight|
|---|---|---|
|A* search|g(n) + h(n)|W = 1|
|Uniform-cost search|g(n)|W = 0|
|Greedy best-first search|h(n)|W = ∞|
|Weighted A* search|g(n) + W × h(n)|1 < W < ∞|

### 1.3. Heuristics

#### Admissible Heuristics

- $h(n)$ is **admissible** if, $\forall n$, $h(n) \leq h^*(n)$
  - $h^*(n)$ = true cost to reach the goal state from n
- **Never overestimates** cost to reach goal
- Example: $h_{SLD}(n)$ never overestimates the actual road distance (roads are at best straight)
- If $h(n)$ is **admissible**, then A* using TREE-SEARCH is optimal

#### Consistent Heuristics

- $h(n)$ is consistent if, for every node $n$ and every successor $n'$ of $n$ generated by any action $a$,

  $h(n) \leq c(n,n') + h(n')$

- If $h(n)$ is consistent,

  $f(n') = g(n') + h(n')$
  
  $= g(n) + c(n,n') + h(n')$
  
  $\geq g(n) + h(n) = f(n)$

- **$f(n)$ is non-decreasing along any path**
- If $h(n)$ is **consistent**, then A* using GRAPH-SEARCH is optimal

#### Admissible vs Consistent

- Consistent $\Rightarrow$ Admissible
- Admissible $\nRightarrow$ Consistent

### 1.4. Adversarial Search

### 1.5. Local Search and Optimization

### 1.6. Constraint Satisfaction Problem

## 2. Knowledge

### 2.1. Logical Agents

#### 2.1.1. Knowledge-based Agents

#### 2.1.2. Propositional Logic

|Connective|Symbol|Statement|
|---|---|---|
|and|∧|Conjunction|
|or|∨|Disjunction<br>(inclusive or)|
|xor|⊕,⊻|Exclusive or|
|not|~,¬|Negation|
|if...then|→,⇒|Implication<br>(conditional statement)|
|if and only if|↔,⟺|Biconditional|

Reading materials: https://calcworkshop.com/logic/

Truth table generator: https://web.stanford.edu/class/cs103/tools/truth-table-tool/

|Topic|Link|
|---|---|
|Propositional Logic|https://calcworkshop.com/logic/propositional-logic/|
|Logical Implication|https://calcworkshop.com/logic/logical-implication/|
|Logical Equivalence|https://calcworkshop.com/logic/logical-equivalence/|
|Predicate Logic|https://calcworkshop.com/logic/predicate-logic/|
|Rules of Inference|https://calcworkshop.com/logic/rules-inference/|

### 2.2. First-Order Logic

### 2.3. Inference in First-Order Logic

## 3. Uncertainty

### 3.1. Probabilities

### 3.2. Bayes' Rule

### 3.3. Bayesian Networks

### 3.4. Hidden Markov Models

### 3.5. Markov Decision Process

## 4. Machine Learning

### 4.1. Supervised Learning

#### 4.1.1. Naive Bayes Classification

#### 4.1.2. Linear Regression

#### 4.1.3. Support Vector Machines

#### 4.1.4. Decision Trees and Random Forests

### 4.2. Unsupervised Learning

#### 4.2.1. Principal Component Analysis

#### 4.2.2. Manifold Learning

#### 4.2.3. k-Means Clustering

#### 4.2.4. Gaussian Mixture Models

#### 4.2.5. Kernel Density Estimation

## 5. Deep Learning

### 5.1. Neural Networks

### 5.2. Convolutional Neural Network

### 5.3. Recurrent Neural Network

## 6. Language
