# Bayesian Inference in Healthcare

## 📚 Overview
This repository contains a mini learning module for bachelor-level students who are being introduced to Bayesian reasoning through healthcare-themed examples. The content is organised around three guided Jupyter notebooks that progressively build intuition:

1. **01_Monty_Hall_Frequentist_vs_Bayesian.ipynb** – contrasts frequentist simulation and Bayesian updates using the Monty Hall problem.
2. **02_Bayes_demo.ipynb** – walks through a medical screening example to compute post-test probabilities.
3. **03_Bayes_exercise.ipynb** – provides a scaffolded exercise applying Bayes' theorem to opioid addiction risk modelling.

An additional stretch notebook, **04_naive_bayes_opioid_demo.ipynb**, connects the statistical foundations to a hands-on machine learning workflow.

Each notebook mixes short explanations with executable code so that learners can observe the math, run experiments, and reflect on the results.

## 🗂 Repository Structure
```
.
├── 01_Monty_Hall_Frequentist_vs_Bayesian.ipynb
├── 02_Bayes_demo.ipynb
├── 03_Bayes_exercise.ipynb
├── 04_naive_bayes_opioid_demo.ipynb   # optional advanced extension
├── Data/
│   └── opiod_raw_data.csv             # dataset used in the naive Bayes demo
├── environment.yml                    # conda environment specification
└── README.md
```

## 🧑‍🏫 Learning Goals
By completing the three core notebooks you will be able to:
- Explain the conceptual difference between frequentist and Bayesian perspectives on probability.
- Translate real-world healthcare questions into the formal components of Bayes' theorem (prior, likelihood, evidence, posterior).
- Apply Bayes' theorem to interpret diagnostic test results and risk predictions.
- Reflect on how data availability, model assumptions, and class imbalance influence decisions in healthcare contexts.

## 🛠 Prerequisites
- **Software**: Python 3.9 or newer, JupyterLab (or the classic Jupyter Notebook interface), and Git.
- **Background knowledge**: Basic probability (events, conditional probability) and familiarity with Python syntax. All advanced math is explained inline.

## ⚙️ Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-account>/Bayesian-in-HC.git
   cd Bayesian-in-HC
   ```

2. **Create the conda environment** (recommended for consistent package versions)
   ```bash
   conda env create -f environment.yml
   conda activate bayesian-in-hc
   ```
   If you prefer `pip`, manually install the packages listed inside `environment.yml`.

3. **Launch Jupyter**
   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

4. **Open the notebooks** in the order shown below and run each cell sequentially (`Shift + Enter`). Keep an eye on the prompts and markdown explanations—many steps build upon the outputs of earlier cells.

## 📓 Notebook Guide
### 1. 01_Monty_Hall_Frequentist_vs_Bayesian.ipynb
- **Purpose**: Compare frequentist Monte Carlo simulation with exact Bayesian updating using the classic Monty Hall game.
- **What you will do**:
  - Simulate thousands of Monty Hall games to see empirical win rates converge to theoretical probabilities.
  - Visualise convergence via bar charts and running averages.
  - Perform an interactive Bayesian update that models how new information (the host revealing a door) reshapes beliefs.
- **Suggested workflow**:
  1. Run the setup cell to import NumPy and fix the random seed.
  2. Execute simulation cells from top to bottom, observing how increasing the number of trials stabilises estimates.
  3. Use the interactive widgets to change priors/likelihoods and interpret how the posterior responds.
- **Reflection prompts**: After each major plot, pause to articulate why the Frequentist estimate and Bayesian posterior agree despite different reasoning paths.

### 2. 02_Bayes_demo.ipynb
- **Purpose**: Apply Bayes' theorem to a medical screening scenario (breast cancer mammography) and unpack every component of the calculation.
- **What you will do**:
  - Translate narrative statements (prevalence, sensitivity, specificity) into numerical priors and likelihoods.
  - Use a contingency table to track patient counts and false positives/negatives.
  - Compute the posterior probability of truly having cancer given a positive test result.
  - Interpret a probability tree diagram to solidify intuition.
- **Suggested workflow**:
  1. Execute the data setup cells to build the population-level counts.
  2. Follow the guided calculation cells—each variable has inline comments linking back to the story.
  3. Review the markdown summary and tree diagram to confirm your understanding before moving on.
- **Reflection prompts**: Consider how false positive rates affect patient anxiety and downstream testing decisions.

### 3. 03_Bayes_exercise.ipynb
- **Purpose**: Practise Bayes' theorem in a healthcare operations context by estimating opioid addiction risk from limited historical data.
- **What you will do**:
  - Start from partially completed code cells that mirror the prior notebook.
  - Fill in missing calculations (sensitivity, specificity, false positive rate) to complete the Bayes formula.
  - Compute the posterior probability that a patient will develop an addiction when flagged by a predictive model.
  - Use the provided `display(Markdown(...))` cell to report your final answer in percentage form.
- **Suggested workflow**:
  1. Review the scenario assumptions in the opening markdown cell.
  2. Run each code cell, completing TODOs where indicated by inline comments.
  3. Verify intermediate values before executing the final display cell.
- **Reflection prompts**: Discuss how changing the prior addiction rate or model sensitivity would affect prescribing decisions.

### Optional Extension: 04_naive_bayes_opioid_demo.ipynb
- **Purpose**: Bridge from manual Bayes calculations to a machine learning pipeline using scikit-learn's Bernoulli Naive Bayes classifier.
- **Dataset**: `Data/opiod_raw_data.csv` (binarised healthcare features and an opioid disorder flag).
- **What you will do**:
  - Prepare binary features, split data, and train two Naive Bayes models with different priors.
  - Compare evaluation metrics (ROC AUC, Precision-Recall) and study how priors influence calibration.
  - Explore both global and local interpretability via likelihood tables and per-patient explanations.
- **Suggested workflow**: Ensure the dataset path is correct, execute each section sequentially, and adjust thresholds to see the impact on alerts.

## ✅ Tips for Successful Runs
- Execute cells in order—later cells rely on variables created earlier.
- If you restart the kernel, rerun all previous cells to regenerate variables and plots.
- Many plots and outputs rely on `%matplotlib inline`; if figures do not appear, verify that the setup cells executed without errors.
- Use `Kernel > Restart & Run All` (in JupyterLab) to reproduce the complete workflow after making changes.

## 🤝 Contributing
Contributions that enhance clarity, add new teaching activities, or improve dataset documentation are welcome. Please open an issue to discuss ideas before submitting a pull request.

## 📄 License
This project is licensed under the [MIT License](LICENSE).
