Phase 4 — Comparative Analysis & Insights

* Models evaluated:
  - SVM (HOG+HSV): test acc ≈ 0.916, val acc ≈ 0.933
  - Random Forest (HOG+HSV): test acc ≈ 0.882, val acc ≈ 0.925
  - CNN: test acc ≈ 0.504, val acc ≈ 0.530

* Best model on this subset: **SVM (HOG+HSV)** with test accuracy ≈ 0.916.
* CNN performance: test ≈ 0.504, val ≈ 0.530.
* Classical feature-based models (HOG + color) clearly outperform CNN on this small, imbalanced dataset.
* Likely reasons CNN underperforms:
  - limited number of training images per class,
  - many signs are tiny / blurry in the original images,
  - simple CNN trained from scratch (no pretrained backbone).
* Most CNN confusions involve 'other-sign' vs the more specific regulatory / pedestrian classes.
* Future work:
  - stronger data augmentation and class balancing,
  - more epochs & LR scheduling,
  - fine-tuning a pretrained backbone such as ResNet-18.