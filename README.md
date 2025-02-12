# Important Note:

- Since this repository doesn't contain the original dataset that has been worked with, the trained ML and DL models; the codes won't be executable!
- The codes are only serves the purpose of demonstration
- The general code quality and structure is not optimal, since this was a time-limited academic study. It does **not** reflects my understanding of clean code :).

# Firat Metin - Master Thesis: Quality Assessment Sensor Fusion for Unobstrusive Cardiorespiratory Signals

## Aim / Workflow:

- The primary aim of this study is to classify cardiorespiratory signals based on their quality and motion artifact measures and to extract heart rate (HR) and respiratory rate (RR) signals while also compensating errors caused by motion artifacts with **Machine Learning** and **Deep Learning** algortihms. After classification of the artifacts for each signal segment, the segments that are afflicted with motion artifacts are reconstructed using **Gaussian Process Models** and **Multi-Task Gaussian Process Models.**
- The workflow starts with preprocessing the signals which refers to `ma/utils.py`.
- It is followed by a feature engineering step which can be inspected in `ma/qa_ma_sqi.py` where various feature sets for cardiorespiratory signals are calculated and saved.
- `ma/qa_ma_sqi.py` also includes the codes for training & using the ML models. To get an overview on the codes, you can also check `demo_qa_ma.ipynb`.
- `ma/qa_ma_networks.py` includes the codes for MLP, CNN, RNN and LSTM Networks. Usage of these DL networks can be inspected in `demo_qa_ma.ipynb`.
- After the classification tasks, GP Models are trained for reconstruction of motion-artifact afflicted HR & RR signals, using the codes in `ma/signal_recoverer.py`, which has the base abstract class for different classifiers, and `ma/gp.py`, which consists of two types of classifier classes.
- Before the MTGP models, a fusion algorithm for HR & RR signals is employed in `ma/signal_recoverer.py`. After obtaining one unified HR and RR signal for each participant MTGP models are also employed by using the codes in `ma/mtgp.py`.

## Summary Of Results

### Signal Quality Assesment:

- For HR: **80.78%** of total accuracy on the test dataset with RUSBoosted Random Forest model with the cECG Signals.
- For RR: **60.00%** of total accuracy on the test dataset with RUSBoosted Random Forest model with the cECG Signals.

### Motion Artifact Detection

- For HR: **88.80%** of total accuracy on the test dataset with MLP network with the cECG Signals.
- For RR: **92.24%** of total accuracy on the test dataset with RUSBoosted Random Forest model with the MIM Signals.

### Signal Recovery

- For HR: Reduction from **612.50 bpm^2** to **150.30 bpm^2** in the errors using GP Models
