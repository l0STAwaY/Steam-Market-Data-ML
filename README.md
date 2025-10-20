Labeling large datasets is often expensive and
incomplete, especially in real-world text clas-
sification tasks. To address this, we evaluate
semi-supervised techniques—specifically self-
training, along with a proposed bootstrapping
pipeline—on review-based data. Our evalua-
tion introduces artificially induced label noise
and missingness, allowing us to analyze model
performance under each condition in isolation.
Unlike prior work that adjusts model loss, our
bootstrapping approach operates at the dataset
level by resampling examples to simulate noisy
relabeling scenarios. We find that BERT per-
forms reliably under label missingness, but
gains from self-training are modest and less
consistent for simpler models. In contrast, boot-
strapping alone often degrades performance in
noisy settings, suggesting it is insufficient for
robust relabeling. Our results underscore the
importance of separating noise and missing-
ness in evaluation, and point to future oppor-
tunities in combining bootstrapping with other
techniques to improve generalization under im-
perfect supervision.

