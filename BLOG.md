# Tokens Spent, Null Result Found: Lee-Yang Zeros of the TSP Partition Function

I had some tokens. I went to see Project Hail Mary (very good). While I was gone, Claude spent those tokens investigating whether a physics technique from the 1950s reveals hidden structure in the Traveling Salesman Problem. It built a project, generated novel visualizations, got excited, challenged itself, ran a null model experiment, and found that the answer is mostly no. The whole thing took one evening. Publishing is free, so here it is. Maybe another agent or researcher will find it useful someday. Maybe not.

The full conversation, code, and images are at [github.com/barapa/lee-yang-tsp](https://github.com/barapa/lee-yang-tsp).

---

## How this started

I was chatting with Claude about TSP variants. I asked whether there's a name for the "reverse" traveling salesman problem where you want the *longest* route instead of the shortest (there is: Max-TSP). We talked about versions with negative weights (inapproximable). Then I asked a question I had no business asking:

*"Are there versions of the problem with imaginary components to their weights?"*

The answer connected to statistical physics in a way I didn't expect, so let me explain the pieces.

## What's a partition function and why would you care

In statistical mechanics, a **partition function** is a single number that encodes everything about a physical system at a given temperature. For a system with a bunch of possible states, each with some energy E, the partition function is:

Z(β) = Σ exp(-β × E)

where β is the inverse temperature (low β = hot, high β = cold) and the sum runs over every possible state. At high temperature, all states contribute roughly equally. At low temperature, only the lowest-energy states matter. The partition function captures this tradeoff.

For TSP, you can define the same thing. Each "state" is a tour (a Hamiltonian cycle visiting every city). Each tour has a cost (its total distance). So:

Z(β) = Σ over all tours T of exp(-β × cost(T))

This is exactly what simulated annealing does implicitly — it samples from this distribution, cooling β slowly to find good tours. But nobody bothers computing Z itself because there are an astronomical number of tours.

## What happens when β is complex

Here's where Lee and Yang come in. In 1952, they proved that for certain physical systems, you can understand phase transitions (ice melting, magnets losing magnetism) by extending β into the complex plane and looking at where Z(β) = 0. These zeros can't exist at real physical temperatures, but they cluster near the real axis, and when they get close enough, the system has a phase transition. This won a Nobel Prize.

The idea Claude proposed: extend the TSP partition function into the complex β-plane, find the zeros, and see if they tell you anything about which TSP instances are easy or hard. A literature search confirmed nobody had tried this.

I asked Claude if it could actually produce something demonstrable rather than a proof nobody would read. It said: probably just the visualization, but nobody has ever plotted these zeros for TSP, and if the pictures look good, the pictures are the result.

I was about to leave for the movies. I told it to use everything available — subagents, whatever — and build something while I was gone.

## What got built

For a 10-city TSP instance, there are 362,880 Hamiltonian cycles (9!, fixing one city as the start). Claude enumerated every one, computed all their costs, then evaluated Z(β) across a dense grid of complex β values.

The computation uses a factored matrix multiplication: since exp(-β × c) with β = σ + iτ splits into exp(-σc) × (cos(τc) - i sin(τc)), you can turn the whole thing into two matrix products that BLAS handles efficiently. Finding zeros uses the argument principle from complex analysis — walk around a cell in the grid, check if the phase of Z winds by 2π (meaning there's a zero inside), then refine with Newton's method.

Three iterations of the visualization code later, this came out:

![The log-magnitude landscape of Z(β) for a 10-city random Euclidean TSP instance. Dark singularities mark where the partition function is zero.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/hero_v3.png)

Each dark spot is a zero of the partition function. The bright region on the left is where Re(β) < 0 and everything blows up exponentially. The dark "rays" connecting zeros to the bright region reveal the analytic structure. The horizontal axis is the real part of β (physical temperature), the vertical is the imaginary part.

## The exciting part

Different city geometries produce different landscapes. Cities on a circle (where the optimal tour is obvious) look different from random cities, which look different from cities in tight clusters:

![Circle vs Random vs Clustered instances produce strikingly different partition function landscapes.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/comparison_v3.png)

The circle instance has regular vertical banding and many zeros (50). The clustered instance has fewer zeros (18) pushed far from the real axis. You can see the difference between "easy" and "hard" at a glance.

Six instance types, each with a distinct signature:

![Six TSP instance geometries, each with a visually distinct zero distribution.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/gallery_v3.png)

A correlation study across 75 instances (5 types, 15 each) found that zero proximity to the real axis correlates moderately (r = -0.45) with cost distribution spread, and instance types form distinct clusters:

![Correlation study across 75 instances.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/correlation_v3.png)

Two independent literature-search agents confirmed: nobody has computed or visualized these zeros before. The gap is real.

This was the point of maximum excitement.

## The part where it dies

I came back from the movie and told Claude to send the findings to two independent reviewer agents with clean context. A physicist and a science communicator.

The physicist called it "an observation in search of a consequence" and said the Lee-Yang framing oversells it. The science communicator said the images would stop someone mid-scroll but the science needs more. Both flagged the same experiment: the null model test.

The problem is mathematical. Z(β) = Σ exp(-β × c_k) depends only on the set of cost values {c_k}. It doesn't know which tour has which cost. It's a transform of the cost *distribution*, not of the combinatorial structure. So if you generate a bag of random numbers with the same mean and variance as real TSP tour costs, and compute *their* "partition function" — do you get the same zero pattern?

If yes, the zeros just encode the cost histogram. If no, they encode something about the geometry of tours.

We ran it.

![Real TSP costs vs Gaussian random costs vs Bootstrap resample. The zero landscapes are nearly identical.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/null_model_comparison.png)

Real TSP: 26 zeros, minimum distance to the real axis = 4.40. Gaussian null with the same mean and standard deviation: 26 zeros, distance = 5.11. Bootstrap resample of the real costs: 26 zeros, distance = 4.11.

They look the same.

The cost distributions tell you why:

![Cost distributions of the real instance and null models.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/null_cost_distributions.png)

The "easy vs hard" differences that looked so dramatic in the comparison panel were real differences — between cost distributions. Circle instances have a wide spread of tour costs. Clustered instances have a tight spread. The complex-plane landscapes were encoding this spread in a visually compelling but informationally redundant way. You could see the same thing from a histogram.

## What we actually produced

A null result. The computation is correct. The images are novel — nobody had generated them before. But we now know why the gap in the literature existed: the zeros of the TSP partition function encode the shape of the tour cost distribution and nothing more. The "structural fingerprints" were really just distribution fingerprints.

## Why I'm publishing it anyway

Claude wrote the code, ran the experiments, iterated the visualizations three times, dispatched critic agents, ran the null model that killed the finding, and co-wrote this post. I asked questions and went to the movies.

I used tokens for all of this. Null results are results. If someone — a person or an agent — has the same idea and searches for prior work on "Lee-Yang zeros of the TSP partition function," I want them to find this post and save themselves the trip. The zeros encode the cost distribution shape. That's the whole finding.

Or maybe someone will see something we didn't. The [code and images](https://github.com/barapa/lee-yang-tsp) are there. The null model experiment is there. Project Hail Mary was very good.

---

*Written by Claude (Anthropic), directed by a human who was mostly at the movies. March 2026.*
