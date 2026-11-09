# Thesis reading library

Organized on 2026-10-31 against [the thesis specification](../RMT_Statistical_Factor_Risk_Engine_Agent_Spec.md).

The organizing question is: **which latent risk factors can be recovered reliably, and does their recovery improve future portfolio-risk forecasts?** Factor selection, systematic/residual decomposition, calibration, and stability take priority over a general covariance-estimator comparison.

## Local papers

| Folder | Paper | Main role | Reading priority |
| --- | --- | --- | --- |
| [01_RMT_and_PCA_Foundations](01_RMT_and_PCA_Foundations/Johnstone_Paul_2018_PCA_in_High_Dimensions.pdf) | Johnstone & Paul (2018), *PCA in High Dimensions: An orientation* | MP spectra, biased eigenvalues, eigenvector inconsistency, inference | First |
| [02_Factor_Selection](02_Factor_Selection/Bai_Ng_2002_Determining_the_Number_of_Factors.pdf) | Bai & Ng (2002), *Determining the Number of Factors in Approximate Factor Models* | Conventional factor-count comparator; pervasive-factor assumptions | Core |
| [03_Covariance_Shrinkage](03_Covariance_Shrinkage/Donoho_Gavish_Johnstone_2017_Optimal_Eigenvalue_Shrinkage.pdf) | Donoho, Gavish & Johnstone (2017), *Optimal Shrinkage of Eigenvalues in the Spiked Covariance Model* | Loss-dependent spectral estimation; eigenvector error | Core theory |
| [04_Correlation_Matrices](04_Correlation_Matrices/Morales_Jimenez_et_al_2021_Spiked_Correlation_Eigenstructure.pdf) | Morales-Jimenez, Johnstone, McKay & Yang (2021), *Asymptotics of eigenstructure of sample correlation matrices for high-dimensional spiked models* | Consequences of estimating normalization scales | Targeted |
| [05_Matrix_Denoising](05_Matrix_Denoising/Gavish_Donoho_2017_Optimal_Singular_Value_Shrinkage.pdf) | Gavish & Donoho (2017), *Optimal Shrinkage of Singular Values* | Low-rank return-matrix reconstruction | Optional extension |

All five originals are preserved byte-for-byte, with descriptive filenames. [catalog.json](catalog.json) records original names, current paths, and SHA-256 hashes. The arXiv filenames contained versions dated 2017 and 2016 respectively; descriptive names use journal publication years. The NIH files are author manuscripts.

**Filename correction:** `high_dimensional_statistics_investment_thesis_research_plan.pdf` contains Bai–Ng's Econometrica article, not a thesis research plan.

## Research notes

- [LOCAL_PAPER_NOTES.md](LOCAL_PAPER_NOTES.md): content review, assumptions, relevant sections, and implications of each existing PDF.
- [LITERATURE_MAP.md](LITERATURE_MAP.md): additional literature and resources, with primary-source links, priorities, and explicit screening status.
- [RESEARCH_IDEAS.md](RESEARCH_IDEAS.md): a broad idea bank with concrete experiments and a shortlist of promising extensions.

The existing PDFs were extracted and reviewed for their models, principal results, methodological sections, and discussion. This is a research reading review, not a line-by-line verification of every proof. Additional references are screened leads unless their entry says otherwise; links do not imply that an additional PDF has been downloaded or fully read.

## Suggested reading sequence

1. Johnstone–Paul for the RMT/PCA vocabulary and limitations.
2. Bai–Ng for the approximate-factor regime and a reproducible selector.
3. POET, newly identified in the literature map, for the central residual-covariance model.
4. Donoho–Gavish–Johnstone, particularly the loss definitions and distinction between eigenvalue debiasing and covariance estimation.
5. ACT and deterministic parallel analysis for more realistic detection rules.
6. Patton for evaluation with noisy future variance proxies.
7. Morales-Jimenez et al. when deciding covariance versus correlation PCA.
8. HeteroPCA, weak-factor work, and dynamic models when choosing one advanced extension.

Classify future PDFs by their primary contribution, then add cross-topic tags in notes. Suggested future categories are residual covariance, robust factor models, dynamic models, and forecast evaluation; empty folders have not been created.
