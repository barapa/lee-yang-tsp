# I Sent an AI to Do Science While I Watched a Movie. Here's What It Found (and Didn't).

*An honest account of computational novelty, beautiful visualizations, and a null result.*

---

It started with a genuinely idle question I asked Claude: *"Is there a term for the reverse traveling salesman problem where instead of trying to find the shortest path that hits every stop, you're trying to find the longest path?"*

There is — it's called Max-TSP. We talked about how the two problems are computationally equivalent but have different approximation properties (Max-TSP is actually easier to approximate, which is counterintuitive). Then I asked: *"Are there versions of the problem with both negative and positive weights?"* Yes — and those are essentially inapproximable. Then, on a whim: *"Are there versions of the problem with imaginary components to their weights?"*

That's where things got interesting, because complex weights connect to statistical physics.

## The idea

In 1952, T.D. Lee and C.N. Yang published a theorem about phase transitions in magnetic systems. The insight, which later won them a Nobel Prize, was that you could understand a system's critical behavior by looking at where its partition function hits zero in the complex plane. These "Lee-Yang zeros" have been studied for spin models, lattice gases, graph coloring — but never for the Traveling Salesman Problem.

The TSP partition function is:

> Z(beta) = sum over all tours T of exp(-beta * cost(T))

When beta is real, this is just the Boltzmann distribution from simulated annealing. But make beta complex and you get a function on the complex plane, complete with zeros that might tell you something about the problem's structure.

I asked: *"Do you have any incredible speculative ideas in this area? Something that hasn't been discovered yet?"* Claude proposed four speculative research directions. The Lee-Yang zeros idea was the wildest. I asked the obvious follow-up: *"Could you actually do work on this that could yield provable results — like, solve something hard and prove it works?"*

The honest answer was no — not for breaking complexity barriers. I pushed back: *"Are any of your speculative ideas likely to produce something I could easily show works, rather than producing a proof that someone would need to read — which no one would?"*

The answer: probably the visualization. Nobody has ever plotted the Lee-Yang zeros of the TSP partition function. "Like the Mandelbrot set," Claude said. "The math is deep but the picture is what gets attention." If the pictures are striking, the pictures are the result.

That was enough for me. I was about to leave for the movies. I told Claude to use all available tools — subagents, Codex, Gemini, whatever — and make something novel, interesting, and shareable while I went to see Project Hail Mary.

Then I left.

## What got built while I was gone

When I came back, there was a complete Python project. Here's what it does:

For a 10-city TSP instance, there are 362,880 distinct Hamiltonian cycles (that's 9 factorial — you fix one city and permute the rest). The code enumerates every single one, computes their costs, then evaluates Z(beta) across a dense grid of complex beta values. The trick that makes this tractable is factoring the computation into matrix multiplications that hit BLAS — what would be a naive double loop becomes two matrix products that NumPy eats for breakfast.

Finding the zeros uses the argument principle from complex analysis: if you walk around a small square in the complex plane and the phase of Z(beta) winds by 2*pi, there's a zero inside. Then Newton's method pins it down to machine precision.

The code went through three iterations. Version 1 was "too stripey." Version 2 was better. Version 3, with log-magnitude heatmaps and a custom plasma colormap, produced this:

![The log-magnitude landscape of Z(beta) for a 10-city random Euclidean TSP instance. Dark singularities mark zeros of the partition function.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/hero_v3.png)

That's the magnitude landscape of the TSP partition function in the complex plane. The dark singularities are where Z(beta) = 0. The bright region at left is where all tour weights grow exponentially. The structured dark rays emanating from each zero reveal the analytic structure that nobody has visualized before.

It's a genuinely beautiful image. I'll give it that.

## The exciting part

Different TSP instance geometries produce dramatically different landscapes:

![Three panels showing Circle (easy) vs Random (medium) vs Clustered (hard) instances with strikingly different zero patterns.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/comparison_v3.png)

Circle instances — cities evenly spaced on a ring — produce regular vertical bands with many zeros close to the real axis. Random Euclidean instances get diagonal dark rays. Clustered instances push the zeros far away, creating a bright, relatively featureless landscape. You can literally *see* the difference between easy and hard.

The gallery shows this across six instance types:

![Six different TSP instance geometries, each producing a visually distinct zero signature in the complex plane.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/gallery_v3.png)

Each geometry has its own visual fingerprint. This was the point where two independent research agents confirmed: nobody has done this before. The gap in the literature is real. We checked arXiv, SIAM, Springer, Google Scholar. Barvinok's work on partition function zeros comes closest, but he treats graph polynomials and matchings, not cost-weighted TSP.

Then came the correlation study — 75 instances across 5 types, 15 each. The zero distribution quantitatively separates instance types. The cost coefficient of variation shows r = -0.45 correlation with how close zeros get to the real axis. Instance types form distinct clusters.

![75-instance correlation study showing moderate correlation between cost statistics and zero proximity.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/correlation_v3.png)

This was the point of maximum excitement.

## The honest assessment

I came back from the movie (Project Hail Mary was excellent, by the way) and asked: "Give me an honest assessment. Are we blowing smoke?"

I told Claude to dispatch independent reviewer agents with clean context — no self-critique, just explain the finding and let them decide. A physicist and a science communicator.

The physicist: "An observation in search of a consequence. The Lee-Yang framing oversells it. But the computation is non-trivial and the gap is real."

The science communicator: "The hero image genuinely stops someone mid-scroll. This is closer to 'first X-ray of a particular fish species' than 'counting ceiling tiles in room 304.'"

Both flagged the same decisive experiment: **the null model test.**

Here's the problem. Z(beta) is a sum over tours: sum of exp(-beta * cost_k). It depends *only* on the multiset of tour costs. It doesn't care which tour has which cost. It's literally a transform of the cost distribution. So if you generate random numbers with the same statistical properties as real TSP tour costs — same mean, same variance — and compute their "partition function" and its zeros... do you get the same picture?

If yes, the zeros are just a fancy histogram transform. If no, they encode something deeper about combinatorial structure.

## The experiment that killed it

We ran the experiment.

![Real TSP vs Gaussian null model vs Bootstrap resample. The zero landscapes are nearly identical.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/null_model_comparison.png)

Real TSP (seed=42): 26 zeros, minimum distance to the real axis = 4.40.

Gaussian null (same mean and standard deviation, completely random costs): 26 zeros, minimum distance = 5.11.

Bootstrap resample (same empirical distribution, resampled with replacement): 26 zeros, minimum distance = 4.11.

They look almost identical.

The cost distributions explain everything:

![The cost distributions of real TSP instances vs null models, showing the statistical similarity.](https://raw.githubusercontent.com/barapa/lee-yang-tsp/main/output/null_cost_distributions.png)

The dramatic differences between "easy" and "hard" instances? They're real, but they're just reflecting different cost distributions. Circle instances have a wide spread of tour costs (many very different tours). Clustered instances have a tight spread (all tours cost about the same). You could see this directly from a histogram. The complex-plane visualization is a beautiful, technically correct, computationally non-trivial way of looking at... a histogram.

## What this actually is

It's a null result.

The computation is correct. The code works. The images are genuinely novel — nobody has produced them before. The gap in the literature is real, and now we know *why* the gap exists: there isn't much there. The zeros encode the shape of the cost distribution, not the combinatorial structure of the problem. The "fingerprints" that looked so promising are just reflecting what you could see from five lines of numpy and a bar chart.

The physicist reviewer had it right from the start: an observation in search of a consequence.

## Why publish anyway

I used tokens for this. A lot of tokens. Claude wrote the code, ran the experiments, iterated the visualizations, critiqued its own work, ran the null model that killed the finding, and is now writing this post. I asked questions and went to the movies.

And the thing is: null results are results. Somebody, someday, might have the same idea — "what if we look at the Lee-Yang zeros of the TSP partition function?" — and spend actual grant money and graduate student hours on it before discovering what we discovered in an evening: the zeros encode the cost distribution shape, full stop. If this post saves someone that detour, the tokens were worth it.

Or maybe someone smarter than me (or my AI) will see something we missed. Maybe the right framing isn't Lee-Yang at all but something else. The code is there. The images are there. The null model comparison is there. Have at it.

Publishing is free. The tokens are already spent. Project Hail Mary was very good.

---

*The code, all visualizations, and the null model experiment are available in the [repository](https://github.com/barapa/lee-yang-tsp). Built entirely by Claude (Anthropic) with human direction. The human's contribution was asking "but is it real?" at the right moment, and going to the movies at the right moment too.*
