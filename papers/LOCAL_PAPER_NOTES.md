# Review of the five supplied papers

Reviewed 2026-10-31. Page references below refer to printed paper pages or named sections; PDF viewer page numbers can differ. Mathematical expressions should be checked against the PDF before transcription into code: extraction sometimes loses square roots and subscripts.

## 1. Johnstone & Paul (2018): PCA in High Dimensions: An orientation

[Local PDF](01_RMT_and_PCA_Foundations/Johnstone_Paul_2018_PCA_in_High_Dimensions.pdf) · [Published author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC6167023/) · DOI: 10.1109/JPROC.2018.2846730.

**Content.** A broad orientation to high-dimensional PCA. Sections III–V explain spectral spreading, spike bias/phase transitions, and eigenvector inconsistency. Section VI discusses universality beyond Gaussian observations; VII covers covariance estimation and sparse PCA; VIII inference; IX nonasymptotic bounds; X extensions including dependent data and resampling.

**What matters here.** With identity noise and aspect ratio q, the sample bulk ends at (1+sqrt(q))², while a bounded population spike separates at 1+sqrt(q). Above separation, the sample eigenvector can remain inconsistent. Thus a conspicuous eigenvalue alone does not certify a precisely estimated risk exposure. Below separation, ordinary PCA loses directional information in the stated regime.

**Assumptions and limits.** The exposition starts with independent Gaussian observations and later relaxes distributional assumptions under specific conditions. Universality does not license arbitrary volatility clustering or temporal dependence. Sparse eigenvectors are a separate modeling assumption; dense market loadings need not satisfy it. The survey distinguishes bounded spikes from settings where spikes diverge.

**Use in thesis.** Foundations chapter, simulation checks, and interpretation of factor-subspace errors. Read Sections III–V closely before implementing MP selection; consult VI and X when interpreting financial departures. Track subspace projection distances and eigengaps in addition to counts.

**Idea.** Plot count, estimated eigengap, subspace change, and subsequent portfolio loss together. Check whether threshold-near factors have unstable directions even when the count itself is stable. This is an empirical proposal, not a conclusion of the survey.

## 2. Bai & Ng (2002): Determining the Number of Factors in Approximate Factor Models

[Local PDF](02_Factor_Selection/Bai_Ng_2002_Determining_the_Number_of_Factors.pdf) · [Original article record](https://www.jstor.org/stable/2692167). Econometrica 70(1), 191–221.

**Content.** PCA estimation of latent factors in large balanced panels, rates up to a rotation, and consistent selection through panel criteria. Section 2 states assumptions; 3 develops factor estimation; 4 establishes selection consistency; 5 gives PCp and ICp rules; 6 supplies simulations and an asset-return application. The concluding discussion addresses missing-data iteration and further questions.

**Core mechanism.** If V(k) is average squared reconstruction residual over all n×p entries, minimize log V(k)+k g(p,n). For ICp2,

\[
g(p,n)=\frac{p+n}{pn}\log\min(p,n).
\]

ICp1 replaces the logarithm by log[pn/(p+n)]; ICp3 uses log[min(p,n)]/min(p,n). The penalty must vanish while dominating the relevant factor-estimation error rate. PCp rules additionally need a noise-variance estimate; the ICp rules avoid that particular dependence on k_max, although the candidate search range still matters.

**Assumptions and limits.** A fixed number of factors; nondegenerate limiting factor covariance; loadings with normalized cross-product converging to a positive-definite matrix; moment and weak-dependence restrictions. The loading condition yields pervasive factors whose leading covariance eigenvalues grow with p. Some heteroskedasticity and serial/cross-sectional dependence are allowed within these restrictions. This is not the fixed-strength BBP regime. The original moment assumptions are substantive, especially for financial tails.

**Use in thesis.** The conventional comparator for H1. Predeclare ICp2 and k_max, include k=0, and report ICp1 sensitivity if justified. Compute V(k) with the paper's normalization rather than substituting an unexplained covariance trace. Handle V(k)=0 by a declared numerical policy and a sensible rank cap. Preprocessing must use only the trailing window.

**Reading targets.** Printed pp. 196–201 for assumptions, factor convergence, and criteria; simulations for dependence sensitivity. The paper's empirical selection is evidence for its particular sample, not a prescribed factor count for French portfolios.

**Idea.** Compare selectors separately in bounded-spike and pervasive-factor simulations before real data. A disagreement may reflect different statistical targets rather than an implementation error.

## 3. Donoho, Gavish & Johnstone (2017): Optimal Shrinkage of Eigenvalues in the Spiked Covariance Model

[Local PDF](03_Covariance_Shrinkage/Donoho_Gavish_Johnstone_2017_Optimal_Eigenvalue_Shrinkage.pdf) · [arXiv record](https://arxiv.org/abs/1311.0851). Local version: v3, 4 June 2017.

**Content.** Loss-specific eigenvalue shrinkage under a fixed-rank spiked covariance model. A simultaneous two-dimensional block decomposition reduces comparison of population and estimated matrices to tractable blocks. The paper studies 26 losses, provides explicit shrinkers for many, and discusses stronger equivariant optimality for particular losses.

**Central insight.** Inverting the sample-to-population eigenvalue map corrects spike bias but generally does not minimize covariance loss: sample eigenvectors remain misaligned. The optimal adjustment depends on the loss. Sections 1 and 5–6 are directly relevant; 7 handles multiple spikes and technical optimality; 9 unknown common noise scale; 10 contrasts this setting with general-spectrum shrinkage.

**Assumptions and limits.** Independent Gaussian samples in the baseline model; p/n tends to gamma in (0,1]; finitely many fixed, distinct spikes above a flat identity bulk. Section 9 rescales a common noise variance. Heterogeneous residual variances and sparse residual correlations do not meet that flat-bulk model automatically. The paper's optimality claims are asymptotic and estimator-class/loss specific.

**A particularly useful connection to the specification.** Let A be the true covariance and C its forecast. Under zero conditional mean,

\[
E[\log\det C+r^\top C^{-1}r\mid\mathcal F_t]
-\log\det A-p
=\operatorname{tr}(C^{-1}A)-\log\det(C^{-1}A)-p.
\]

This is the paper's **entropy loss** L_ent(A,C)=L_st(C,A), given its Section 5 naming convention. Its Stein loss instead uses A^{-1}C. The argument order matters. The identity follows directly by taking the conditional expectation; it does not establish that the paper's shrinker is optimal for changing financial covariances.

**Use in thesis.** Explain why a Frobenius-optimal estimator need not best serve the declared forecast score. Keep general-spectrum Ledoit–Wolf nonlinear shrinkage distinct from this finite-spike family. Do not implement all 26 losses for the core thesis.

**Idea.** A controlled synthetic comparison of eigenvalue inversion, Frobenius shrinkage, and entropy-oriented shrinkage, evaluated on independent future observations. Use this as a mechanism illustration, then test only a justified reference on financial data.

## 4. Morales-Jimenez, Johnstone, McKay & Yang (2021): Asymptotics of eigenstructure of sample correlation matrices for high-dimensional spiked models

[Local PDF](04_Correlation_Matrices/Morales_Jimenez_et_al_2021_Spiked_Correlation_Eigenstructure.pdf). Statistica Sinica 31(2), 571–601; DOI: 10.5705/ss.202019.0052.

**Content.** First-order limits and fluctuation distributions for spiked sample correlation eigenvalues and eigenvectors. Sections 1–2 state the model and principal theorems; later sections establish normalized quadratic-form results and eigenvalue/eigenvector proofs.

**What changes after standardization.** First-order spiked limits match the corresponding covariance problem under the paper's model, but fluctuations include extra normalization terms and can differ substantially. Correlation normalization often lowers fluctuations in the examples, but the paper also provides a counterexample to universal variance reduction.

**Important restriction.** Model M has a fixed-dimensional signal block independent of an increasing iid noise block, iid observations, and a finite (4+delta)-moment condition. Its block correlation structure is much more specific than a dense financial approximate-factor model. Do not cite the theorems as an automatic guarantee for standardized French returns.

**Use in thesis.** The covariance-versus-correlation decision is statistically consequential, not cosmetic. For correlation fitting, explicitly reconstruct the forecast in return units, C=D R_hat D, with training-only estimated scales D. This transformation also changes loading coordinates; Euclidean principal angles in standardized coordinates and return coordinates answer different questions.

**Reading targets.** Model M and Theorems 1–3; the variance-reduction condition and its failure example in Section 2; Gaussian examples demonstrating normalization effects. Check formulas visually because extracted radicals are imperfect.

**Idea.** Hold residual method and factor count fixed, compare covariance and correlation extraction, then separately vary the selector. This separates effects of normalization from effects of counting factors.

## 5. Gavish & Donoho (2017): Optimal Shrinkage of Singular Values

[Local PDF](05_Matrix_Denoising/Gavish_Donoho_2017_Optimal_Singular_Value_Shrinkage.pdf) · [arXiv record](https://arxiv.org/abs/1405.7511). Local version: v3, 15 May 2016; IEEE Transactions on Information Theory 63(4), 2137–2152 (2017).

**Content.** Estimate a low-rank rectangular signal X from Y=X+sigma Z by shrinking singular values. Losses include Frobenius, operator, nuclear and Schatten norms. The proof strategy again uses small blocks and asymptotic singular-vector overlaps. Section 8 handles noise rescaling, unknown noise estimation and general white noise under additional conditions; 9 studies finite simulations.

**Distinction from the covariance paper.** The target is a signal matrix, not its covariance or an inverse covariance. A return-panel denoiser is not by itself a complete positive-definite future risk model. Squaring its shrunken singular values does not automatically yield an optimal covariance forecast.

**Assumptions and limits.** Fixed signal rank and fixed signal singular-value strengths in the normalized proportional-growth framework. The core model uses orthogonally invariant white noise; the general-iid extension uses a generic random signal orientation and moment conditions. Real financial residuals can be heteroskedastic, cross-correlated, dependent, and heavy tailed.

**Use in thesis.** Optional common-component regularization and a conceptual bridge to regularized factor estimation. Read Sections 1–3 for model and scaling, 4–6 for loss-specific shrinkage, and 8 before noise estimation. Keep singular-value units and sample-covariance eigenvalue units separate.

**Idea.** With selected factor directions held fixed, compare unshrunk and shrunk common-component amplitudes. Re-estimate residual risk consistently and account for any factor/residual cross terms or define a coherent covariance estimator explicitly. Simply shrinking fitted returns and then adding their residual covariance can break the original PCA orthogonality decomposition.

## Shared implications

1. Factor count, common-component reconstruction, and covariance forecasting have different statistical objectives.
2. Bounded spikes, pervasive factors, and general population spectra require different theoretical references.
3. Standardizing total asset volatility does not whiten idiosyncratic noise.
4. The most conspicuous gap in the supplied library is a paper directly supporting sparse residual covariance. POET should be the next core read.
5. The evaluation chapter needs its own literature: spectral optimality cannot substitute for future risk calibration and uncertainty estimates.
