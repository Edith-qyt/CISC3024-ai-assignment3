# Stage 1 - PCA/LDA Algorithm Research and Selection

**Course:** CISC3024 Pattern Recognition  
**Research cutoff:** 17 November 2026  
**Scope:** algorithm research and selection only. No code, packages, model weights, or datasets were created, installed, or downloaded.

## Executive summary

Three candidates met the minimum evidence threshold for a technical comparison. Two are directly about image analysis: ATAM-PCA (2023) for image recognition and JSLLDA (2024) for hyperspectral image classification. KPCA-CAM (2024) applies kernel PCA to explainability of CNN vision models. PCA/LDA is central in each candidate, not merely a baseline. A combined PCA-then-LDA pipeline is not itself the contribution of these candidates.

**Provisional recommendation:** select ATAM-PCA if the course expects a recognizer whose primary method is an improved PCA. It best matches the assignment wording, but its paper is paywalled and I did not verify author code or exact datasets from an accessible primary full text. Select JSLLDA if open access and a modern LDA method matter more; confirm its dataset details before approval. KPCA-CAM has a clear open preprint and benchmark description, but reported evaluation on ILSVRC/ImageNet makes exact reproduction impractical for a typical student machine unless the scope is explicitly reduced.

The links below distinguish DOI/publisher or arXiv primary records from search and metadata indexes. OpenAlex was used as a discovery and cross-checking index, not treated as a substitute for the paper. No official repository was verified for any of the three. This is a negative search result, not proof that no repository exists.

## Candidate papers

### 1. ATAM-PCA (improved PCA for image recognition)

- **Paper:** Pengfei Bi and Xue Du, “Arbitrary Triangle Structure Adaptive Mean PCA and Image Recognition,” *IEEE Transactions on Circuits and Systems for Video Technology*, 2023. [DOI / IEEE record](https://doi.org/10.1109/TCSVT.2023.3289716)
- **Evidence checked:** OpenAlex record confirms title, authors, year, journal, DOI, and abstract. The abstract describes an image-recognition method, explicitly names Arbitrary Triangle Adaptive Mean PCA (ATAM-PCA), and says it uses a flexible l2,p norm, adaptive mean/centralization, and an iterative solver. It claims experiments on several databases, but the abstract does not name them. The record reports the article as closed access; I could not verify detailed experimental settings against the full text.
- **Why PCA is core:** the proposed algorithm is an alteration of PCA’s objective/mean and projection structure, not a downstream application that happens to use PCA.
- **Repository:** no author-linked or otherwise official code repository verified. Search-result repositories for generic LDA/PCA implementations are not this paper's code and are excluded.
- **Data, dependencies, compute, weights:** exact benchmark names, data access, dependencies, runtime, hardware, and pretrained weights remain **unverified** until the full paper or an author project page is available. The method abstract calls the solver fast and iterative, but that is not enough evidence to claim a hardware requirement.
- **Student feasibility:** likely feasible to implement from equations on a small image-recognition benchmark if the full method is obtained. Exact replication is currently **medium/low confidence** because dataset identities, parameter settings, and official implementation were not verified. Image dimensions and sample count should be capped for CPU-only tests.
- **Main risks:** paper access; ambiguity in the flexible norm and convergence/initialization details if only the abstract is available; unverified datasets and code; fair comparison requires consistent train/test splits and reporting solver convergence as well as accuracy.

### 2. JSLLDA (sparse local LDA for hyperspectral image classification)

- **Paper:** Cong-Yin Cao, Meng-Ting Li, Yang-Jun Deng, Longfei Ren, Yi Liu, and Xinghui Zhu, “Joint Sparse Local Linear Discriminant Analysis for Feature Dimensionality Reduction of Hyperspectral Images,” *Remote Sensing*, 16(22), 4287, 2024. [DOI / open publisher record](https://doi.org/10.3390/rs16224287) | [Publisher PDF](https://www.mdpi.com/2072-4292/16/22/4287/pdf?version=1731922814)
- **Evidence checked:** OpenAlex metadata gives all six names, year, journal, DOI, open-access status, and abstract; the PDF is linked from that record. The abstract states experiments on three datasets and comparison to related methods. The accessible metadata did not expose the dataset names or linked repository, and direct publisher-page retrieval was denied in this research environment. Details not visible in the abstract are not inferred here.
- **Why LDA is core:** JSLLDA is explicitly an LDA-based feature dimensionality reduction method. The abstract describes embedding regression/locality-preserving regularization, a row-sparse projection with an L2,1 constraint, reconstruction-error handling for occlusion, and an orthogonality constraint.
- **Repository:** no official code link verified. The open-access paper is a valuable implementation source, but does not establish that code is public.
- **Data, dependencies, compute, weights:** the abstract confirms three hyperspectral datasets but does not name them in the verified view. Data sizes, download sources, library versions, runtime/hardware, and pretrained models are **unverified**. Classical subspace optimization should not inherently require a GPU; this is an engineering expectation, not a claim reported by the authors. High spectral dimensionality and iterative matrix operations may still dominate memory/time.
- **Student feasibility:** promising if the paper's named benchmarks and preprocessing are confirmed. Small public hyperspectral benchmark cubes are often manageable locally, but the assignment should not commit to a specific dataset until it is confirmed from the paper. A CPU reproduction with reduced parameter sweeps is plausible; exact scaling remains unknown.
- **Main risks:** optimization details and stopping criteria; supervised labels and train/test split protocol; high-dimensional covariance/matrix costs; hyperspectral datasets can have spatial leakage if pixels are randomly split. Prefer spatially disjoint splits where appropriate and match the paper's protocol for the primary comparison.

### 3. KPCA-CAM (kernel PCA for vision-model explainability)

- **Paper:** Sachin Karmani, Thanushon Sivakaran, Gaurav Prasad, Mehmet Ali, Wenbo Yang, and Sheyang Tang, “KPCA-CAM: Visual Explainability of Deep Computer Vision Models Using Kernel PCA,” *2024 IEEE 26th International Workshop on Multimedia Signal Processing (MMSP)*, 2024. [DOI / IEEE record](https://doi.org/10.1109/MMSP61759.2024.10743968) | [arXiv paper and abstract](https://arxiv.org/abs/2410.00267)
- **Evidence checked:** arXiv record identifies all six authors, 2024 submission, CV category, and the IEEE MMSP 2024 journal reference. Its abstract says KPCA-CAM uses PCA with a kernel trick on CNN activations to derive improved class activation maps, and reports evaluation on ILSVRC across different CNN models. OpenAlex independently records the same conference and DOI. Thus PCA is the central proposed attribution method, while CNNs supply the evaluated vision models.
- **Repository:** no official code repository was linked in the verified arXiv metadata/abstract. Repository existence remains **unverified**; do not assume one from the paper title or third-party forks.
- **Data/dependencies:** ILSVRC (ImageNet Large Scale Visual Recognition Challenge) is explicitly named by the abstract. An exact reproduction likely needs image data plus compatible CNN architectures/checkpoints and a deep-learning stack; the paper's exact versions and checkpoints were not verified here. No pretrained checkpoint link was confirmed from the metadata view.
- **Compute/feasibility:** PCA/KPCA over activation maps is accessible in principle, but reproducing all ILSVRC/model experiments is high-cost in storage and compute. A small, explicitly labeled subset with one public pretrained CNN could support a controlled qualitative/quantitative demonstration without claiming full reproduction. GPU is useful for CNN inference but is not necessarily required for a small subset; this is an estimate, not an author-specified requirement.
- **Main risks:** kernel choice and scaling; high memory cost of kernel matrices if sample count is not controlled; dependency/checkpoint compatibility; CAM evaluation metric and data subset may not match the paper. Explainability is less directly a recognition classifier than the course's wording may imply.

## Comparison

| Candidate | Year / venue | PCA or LDA role | Vision task | Code / data evidence | Student feasibility |
|---|---|---|---|---|---|
| ATAM-PCA | 2023, IEEE TCSVT | Improved PCA is the proposed method | Image recognition | Official code not verified; abstract says several unnamed databases; paper closed in metadata | Medium if full paper can be obtained; exact reproduction uncertain |
| JSLLDA | 2024, *Remote Sensing* | Sparse/local LDA is the proposed feature-reduction method | Hyperspectral image classification | Open paper; abstract confirms three datasets but verified names/code not recovered | Medium-high after confirming data and protocol; CPU likely possible with bounded runs |
| KPCA-CAM | 2024, IEEE MMSP | Kernel PCA on CNN activations is central to attribution | CNN explainability for classification/object detection | Open arXiv; ILSVRC named; no official code/checkpoint verified | Low for full reproduction; medium for a clearly scoped subset |

## Ranked recommendations

1. **ATAM-PCA - best assignment fit, conditional on obtaining the full paper.** It is a recent, improved PCA method whose stated application is image recognition, so the algorithm and evaluation story align tightly with a PCA/LDA pattern-recognition assignment. It is ranked first for topical fit, not reproducibility: exact datasets, settings, and code are still missing from verified evidence. Before approval, obtain the full text and confirm a small benchmark plus all implementation equations.
2. **JSLLDA - best open-paper/reproducibility balance, conditional on dataset verification.** It is a recent LDA method with a clear computer-vision-adjacent classification task and an open publisher PDF. Its sparsity/locality/robustness mechanisms make a meaningful experiment possible. It can move to first place if its dataset names, splits, and optimization details in the full paper are sufficiently explicit. The current report does not assert that author code is available.
3. **KPCA-CAM - strongest public description, weaker fit and full-scale feasibility.** The arXiv record gives a directly accessible paper and names ILSVRC, and kernel PCA is clearly central. However it evaluates explanations of deep models rather than a stand-alone recognition projection/classifier, and full ILSVRC reproduction has substantial data/model costs. Prefer only if the instructor accepts explainability as the application and a limited replication scope.

No method is recommended solely because it is newest. Each recommendation depends on paper-level detail and a public-data plan being confirmed before Stage 2.

## Experiment planning considerations (not execution)

- For an approved classical PCA/LDA candidate, use a small benchmark with fixed, documented splits; compare against the corresponding classical PCA or LDA baseline and a simple classifier; report accuracy plus runtime and feature dimension. Add repeated seeds or standard deviations when the split is stochastic.
- For face/image recognition, control image resolution, class count, and samples per class; all fitted transforms and normalization must use training data only to prevent leakage.
- For hyperspectral classification, preserve the spatial structure and avoid random pixel leakage; document label sampling and class imbalance.
- For KPCA-CAM, a subset experiment must be described as a scoped reproduction, not as validation of the ILSVRC-wide reported result. Choose the classifier/CNN, checkpoint, image subset, kernel, and attribution comparison only after checking the paper's methodology.
- Record environment and hardware before experiments. No packages or data have been selected or installed at this stage.

## Source and verification notes

1. Assignment brief: local `AIassign3.pdf`; it requires AI-assisted algorithm search, implementation, experiment settings/results, learning reflection, and a source-code webpage link in the eventual report, and says students may not write code themselves. This stage does not execute those later steps.
2. ATAM-PCA bibliographic and abstract cross-check: [IEEE DOI](https://doi.org/10.1109/TCSVT.2023.3289716) and OpenAlex discovery API (query title: “Arbitrary Triangle Structure Adaptive Mean PCA and Image Recognition”). I verified the title, authors, venue, year and abstract in the index; detailed datasets and repository were not verified.
3. JSLLDA: [MDPI article](https://doi.org/10.3390/rs16224287), [publisher PDF](https://www.mdpi.com/2072-4292/16/22/4287/pdf?version=1731922814), and OpenAlex discovery API (query DOI: 10.3390/rs16224287). Metadata/abstract were checked; direct article retrieval was blocked in this environment, so details absent from the metadata are marked unknown.
4. KPCA-CAM: [arXiv abstract](https://arxiv.org/abs/2410.00267), [arXiv PDF](https://arxiv.org/pdf/2410.00267), and [IEEE DOI](https://doi.org/10.1109/MMSP61759.2024.10743968). Authors, abstract, category, date and venue reference were verified in the arXiv record; code availability and detailed dependency/hardware information were not.
5. Other indexed leads considered but not recommended: “Covariance Regularization for Probabilistic Linear Discriminant Analysis” (Zhiyuan Peng et al., ICASSP 2023, [DOI](https://doi.org/10.1109/ICASSP49357.2023.10094608)) is a genuine PLDA variant, but the abstract's primary application is speaker verification rather than visual recognition; “GO-LDA: Generalised Optimal Linear Discriminant Analysis” (2023 arXiv, [record](https://arxiv.org/abs/2305.14568)) is an algorithmic lead whose direct visual-pattern-recognition fit and reproducibility were not established enough here. These were excluded from the top three rather than overstate their fit.

**Verification boundary:** Publication facts and abstract-level mechanisms are supported by primary DOI/arXiv records plus bibliographic cross-checks. Dataset names beyond the explicit ILSVRC mention, official repository status, specific dependencies, hardware, and checkpoint availability are explicitly unresolved unless stated otherwise. Recommendations are feasibility judgments, not reproduced results.
