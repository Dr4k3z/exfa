# Research ideas for the statistical factor risk engine

Prepared 2026-10-31 from the thesis specification, the supplied PDFs, and [the additional literature](LITERATURE_MAP.md). These are hypotheses and experimental proposals, not findings, novelty claims, or commitments to implement everything. Breadth is intentional at this stage; a thesis should eventually choose one advanced extension.

## Most promising directions

| Direction | Why it fits | Smallest convincing experiment | Priority |
| --- | --- | --- | --- |
| Detectable factors versus risk-useful factors | Connects H1 directly to the primary application | Factor-count rules with one fixed residual method; evaluate future QL and portfolio variance scores | Core |
| Heterogeneous-noise-aware selection | Addresses a central failure of a single MP edge | Classical MP versus ACT or DPA, with causal preprocessing and matched residual estimates | Strong extension |
| Factor count and residual dependence jointly | Underfitting can masquerade as dense residual covariance | Small predeclared selector × residual-method design | Core H1/H2 |
| Persistence beyond mechanical window overlap | Turns H3 into a falsifiable question | Compare overlapping and disjoint-window subspace stability, then test a causal persistence rule | Strong extension |
| Separate volatility timing from correlation structure | Explains why an elegant static model can forecast badly | Apply the same causal volatility model to alternative correlation/factor estimators | Strong diagnostic |
| Loss-aware spectral estimation | Uses the supplied shrinkage paper meaningfully | Known-covariance simulation scored by future QL and fixed-portfolio variance losses | Supporting theory |

## A. Factor detection and recovery

**1. Separate three factor counts.** Distinguish the simulated true dimension, a selector's detectable dimension, and the dimension giving the best validation risk score. Fix the final test period. On financial data only the latter two are observable; do not label a test-optimal count as truth. Compare curves over a small k-grid, then evaluate locked selection rules. Sources: R01–R05.

**2. Diagnose single-scale MP failure.** Simulate no common factors with unequal diagonal residual variances. Measure false selections under different variance dispersions. Repeat with known versus estimated scaling and with a real low-rank component. This isolates noise-scale misspecification from genuine factors. Sources: R02, R09.

**3. Compare covariance and correlation PCA.** First fix k and the residual method, change only normalization, and reconstruct in return units. Then compare selectors. Use equal-weight and concentrated portfolios to test which risks each representation captures. Distances between subspaces must specify their coordinates. Local correlation paper; R02.

**4. Measure market-factor shadowing.** Hold weaker signals fixed while increasing the dominant signal. Test whether weaker factors disappear under the selected rule, and whether a justified deflation method helps. Deflation changes residual noise properties; check the method's assumptions instead of iterating the original MP edge blindly. Source: R01.

**5. Map detection against recovery quality.** In simulations vary spike strength and p/n. Record selection probability, projection error and held-out risk loss separately. Include a neighborhood of the phase transition rather than only obvious strong signals. Sources: local PCA survey, R21–R23.

**6. Bridge strength regimes explicitly.** Use loadings whose squared norm scales as p^alpha for alpha=0, intermediate values, and 1, with noise fixed. Vary p and n together and report finite-sample limitations. This makes bounded-spike versus pervasive-factor distinctions visible. Sources: R05, R23.

**7. Calibrate a financial null.** Within training data, explore independent column-wise block resampling or circular shifts to weaken contemporaneous cross-asset alignment while retaining some univariate behavior. Such schemes also alter common volatility and dependence. Validate their null on synthetic examples and report what is preserved; do not present them as universally valid spectral tests. This is a proposed method requiring its own justification.

**8. Examine localization.** Track inverse participation ratios of eigenvectors. A large eigenvalue localized in a few assets may reflect a concentrated group rather than a pervasive market factor. Relate localization to industry/size groups and future risk of concentrated portfolios. It is a diagnostic, not an automatic deletion criterion. Source: R25; local PCA survey.

## B. Factor/residual structure

**9. Let residual modeling expose underfitting.** For a small set of k values, compare diagonal and POET-style residual estimates. Track remaining leading residual eigenvalues, off-diagonal dependence and forecast scores. An omitted common factor may appear as dense residual structure. The aim is to explain the interaction, not select winners over a large grid. Source: R06.

**10. Compare adaptive and uniform thresholds.** With factors fixed, test whether entry-specific residual thresholds improve forecasts when residual scales differ. Record sparsity and conditioning as diagnostics. Use validation-only tuning and document stabilization. Sources: R06, R07.

**11. Test whether sparsity is plausible for aggregated portfolios.** French portfolios can share economic exposures, and some universes overlap in underlying holdings. Their residual covariance need not resemble that of individual equities. Compare residual-density and sector patterns across the 100 portfolios and 49 industries separately before drawing sparsity conclusions. Source: R06; D01.

**12. Explore group structure before free-form complexity.** A sector-block residual model may sit between diagonal residuals and unrestricted sparse residuals. Use declared industry groupings, or learn groups only in training. Test whether a few group covariances explain improvements. This is an optional proposed model, not established POET theory.

**13. Quantify the cost of positive-definite repair.** Log the repair size, affected eigenvalues and resulting changes in each portfolio's forecast variance. If an apparent gain is driven by aggressive repair, the regularizer is part of the model and must be reported. Compare an explicitly defined diagonal loading policy on matched fits.

**14. Use a frozen-subspace experiment.** Estimate directions on one past block and amplitudes/residuals on a later block available before the forecast. Compare with ordinary fitting to test whether direction estimation and risk calibration benefit from separate samples. The shorter effective sample and regime drift may outweigh any reduction in overfitting. Source inspiration: R15.

**15. Compare factor decomposition with full-spectrum cleaning.** Compare one structured model to one whole-covariance nonlinear-shrinkage reference on QL and portfolio scores. Evaluate interpretability, subspace behavior and systematic/residual attribution only for models that define them. A spectral reference need not receive an arbitrary factor decomposition. Sources: R12–R14, R18.

**16. Study robust factors on extreme days.** Compare a robust elliptical-factor approach with ordinary PCA/POET under controlled heavy tails, then on real returns. Track both bulk performance and predeclared stress periods. Check moment assumptions and scatter-to-covariance scale identification. Source: R08.

## C. Stability and dynamics

**17. Separate overlap from persistence.** Adjacent 252-day windows share almost every observation; their similar subspaces can be mechanical. Compare overlap-adjusted or disjoint historical windows and simulate constant versus changing loadings. Do not interpret raw adjacent-window similarity as independent evidence for persistence. Source inspiration: R25; proposed diagnostic.

**18. Test causal count hysteresis.** Require repeated historical evidence before adding a factor, and a separately defined removal rule. Compare against instantaneous selection under identical residual modeling. Tune patience and thresholds only on validation. Measure both lower count turnover and delayed recognition of genuine changes. This is a proposed rule, not a verified theorem.

**19. Compare subspaces rather than named PCs.** Near-equal eigenvalues can swap or rotate without a meaningful change in common-risk span. Track projection matrices and principal angles. Use Procrustes alignment for displays while keeping inference rotation aware. Separate dimension changes from within-dimension rotation.

**20. Study attribution stability.** Measure changes in total systematic and residual risk, then coordinate-dependent factor contributions under a declared alignment. A smoother factor naming scheme must not be mistaken for a more accurate forecast. Compare attribution changes with subsequent portfolio variance errors.

**21. Separate volatility timing and correlation cleaning.** Write C_t=D_t R_t D_t. Hold a causal volatility forecast D_t common across models while varying factor/correlation estimation; then hold R_t fixed while changing D_t. This tests whether poor calibration is primarily stale volatility. Treat causal intrawindow standardization and forecast-day rescaling as distinct choices. Source: R17.

**22. Study weighting with the right null.** Compare uniform windows and exponential weighting with the same factor/residual strategy. If using spectral detection, use a justified weighted noise model. Report half-life and effective sample size as diagnostics; an effective-size substitution in MP is only a heuristic unless established. Source: R16.

**23. Separate loading changes from factor-volatility changes.** In simulations switch B with factor covariance fixed, then switch factor covariance with B fixed. Compare subspace alarms, count changes and future risk errors. A single crisis-period plot cannot identify which mechanism caused a breakdown.

**24. Contrast the two window lengths mechanistically.** The 504-day window reduces sampling noise but may mix regimes; 252 days adapts faster. Relate performance differences to factor strength and drift in simulations, then assess financial sensitivity without selecting the best window on the final test.

## D. Risk forecasts and evaluation

**25. Add portfolio-level QLIKE.** For forecast v_t>0 and next-day squared portfolio return y_t, use log(v_t)+y_t/v_t, up to constants. It remains defined at y_t=0. Combine it with calibration ratios: a model can be calibrated on average and still time risk poorly. Keep full-matrix QL as the specification's primary score. Source: R19.

**26. Avoid a market-only evaluation.** Equal-weight portfolios may be dominated by the first factor. Add a fixed seeded portfolio ensemble and a small prespecified concentrated/group set to test weaker directions. Distinguish long-only random weights from signed spreads. Report ensemble uncertainty without treating portfolios observed on the same day as independent replications.

**27. Test neutral portfolios carefully.** Construct a portfolio neutral to estimated leading factors using only past data and explicit leverage constraints. Its future risk is sensitive to residual covariance and exposure error. Compare on a common portfolio set for forecast scoring; model-specific optimized portfolios answer a different decision question.

**28. Study systematic/residual forecast calibration cautiously.** Those parts are latent, so total realized portfolio variance does not directly verify their separate correctness. Use simulation for attribution truth. On real data freeze fitted coordinates and specify an out-of-sample projection diagnostic, including cross terms, rather than pretending the decomposition is observable.

**29. Decompose synthetic error sources.** With known B, factor covariance and residual covariance, alternate oracle and estimated pieces to isolate selection, subspace, amplitude and residual errors. These counterfactuals are simulation tools, not implementable real-data forecasts. Their purpose is to explain why a modeling choice changes risk loss.

**30. Match multi-day targets precisely.** Compare h times the one-day variance with the squared h-day cumulative portfolio return only under a clearly stated forecast and portfolio-holding convention. Summed daily squared returns are a different target. Include serial covariance terms if the horizon model needs them; account for overlapping outcomes.

**31. Use paired loss uncertainty.** Compare models on common dates and calculate intervals from the loss differential with an appropriate dependence-aware method. Prespecify resampling blocks or HAC bandwidth policy, and account for overlapping horizons. Report effect sizes, not just rejection decisions. Source: R20.

**32. Keep stress analysis falsifiable.** Define crisis dates externally or define causal regime indicators from past information; choose the policy before final testing. Compare calibration within regimes and include counts/intervals. Retrospective crisis explanations should be labeled descriptive.

**33. Study model averaging as uncertainty control.** Average positive-definite covariance forecasts from a small prespecified set of neighboring counts using training/validation-derived weights. Compare the mixture with hard selection. PD is preserved by nonnegative weights summing to one; factor attribution of the mixture needs its own definition. Proposed extension.

**34. Check mean and scale sensitivity.** The covariance QL expression scores covariance under a zero conditional-mean convention. If centering/mean estimation changes, distinguish centered forecast errors from raw returns. Report a small controlled sensitivity check so improvements are not merely inconsistent preprocessing.

**35. Test approximate spanning without naming factors prematurely.** Regress estimated training-period factor returns on observed French factors or compare subspaces, then assess the relationship on a later block. Alignment can aid interpretation but does not prove causal identification or price-of-risk relevance. D01; interpretation extension.

**36. Audit historical data vintages.** Freeze French downloads and retain metadata; note that current historical files can be revised. A cutoff-safe algorithm applied to revised history is still a retrospective experiment, not proof of a historical point-in-time trading result. Source: D01.

## A manageable first empirical program

1. Build the specification's complete baseline with fixed k, positive diagonal residual covariance, equal-weight risk and covariance QL.
2. For H1, compare fixed k, a documented MP rule and Bai–Ng with residual modeling held fixed. Add ACT or DPA only as a justified extension.
3. For H2, fix selected factors and compare diagonal with sparse residual covariance. Record all numerical repairs.
4. Add a single whole-covariance shrinkage reference to assess whether structure helps forecast quality.
5. For H3, choose either persistence, heterogeneous-noise handling, or volatility dynamics based on the weakness observed on training/validation data.
6. Lock configurations and evaluate the untouched period once. Preserve negative results.

The clearest possible contribution is evidence about **when recovering additional factors improves future risk estimation, and when residual modeling or volatility adaptation matters more**. Novelty must be assessed against the closely related factor/dynamic-covariance work, especially R18, before being claimed.
