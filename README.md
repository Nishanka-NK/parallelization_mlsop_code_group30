PARALLELIZATION AND MULTI-CORE TRAINING

# ML System Optimization: Parallelization and Multi-Core Training

This project demonstrates the optimization of a machine learning system using pandas and scikit-learn, focusing on parallelization (multi-core) for improved speedup and efficiency. The experiment compares single-core and multi-core Random Forest training on a large synthetic dataset.

---

## Problem Formulation

**Objective:**
- Optimize a Random Forest classifier for large-scale data using parallel (multi-core) training.
- Evaluate speedup, response time, and accuracy.

---

## Design

- Algorithm: Random Forest (scikit-learn)
- Parallelization:
  - Single-core: n_jobs=1
  - Multi-core: n_jobs=-1 (uses all available CPU cores)
- Data Handling: pandas DataFrame
- Train/Test Split: scikit-learn's train_test_split
- Performance Metrics: Training time, accuracy

---

## Implementation

See the notebook `ML_System_Optimization.ipynb` for:
- Data generation (synthetic, 100,000 samples, 20 features)
- Single-core and multi-core Random Forest training
- Timing and accuracy measurement
- Comparison graphs
- Analysis and discussion

---

## Results & Discussion

- **Training Time:** Multi-core training significantly reduces model fitting time compared to single-core.
- **Accuracy:** Both approaches achieve similar accuracy, showing parallelization does not compromise model quality.
- **Scalability:** Multi-core parallelism is efficient for large datasets on a single machine.
- **Analysis:** See notebook for detailed graphs and discussion.

---

## Conclusion

Parallelization using scikit-learn's n_jobs parameter enables efficient and scalable machine learning on modern hardware. For even larger data or distributed settings, frameworks like Spark MLlib or Dask may be considered.

---


