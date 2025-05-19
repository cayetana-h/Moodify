# Moodify

# Spotify Mood Classification

This project explores how machine learning can simulate human perception of music mood by analyzing audio features from over 230,000 Spotify tracks. Since the dataset contains no explicit mood labels, we generate them using unsupervised clustering, and then train supervised models to predict those moods from audio features.

The goal is to bridge the gap between the **emotional experience of listening to music** and the **quantitative data** that a machine can understand.

---

## How It Works

1. **Dataset**  
   - 232,000+ Spotify tracks with audio features from the Spotify API  
   - Features include: valence, energy, tempo, danceability, loudness, etc.

2. **Clustering for Mood Labels**  
   - Mood labels are generated using **KMeans** (n=5 clusters) on selected audio features  
   - These labels represent pseudo-emotional "moods"

3. **Preprocessing + Feature Engineering**  
   - One-hot encoding for categorical features (`key`, `mode`, `time_signature`)  
   - Feature scaling with `StandardScaler`  
   - Polynomial & interaction features created (e.g., `danceability * loudness`)

4. **Supervised Classification**  
   - Models trained to predict mood cluster:  
     - Logistic Regression (baseline)  
     - MLP (neural net)  
     - Random Forest  
     - XGBoost  
     - Gradient Boost
 

5. **Evaluation**  
   - Accuracy, macroF1, per-class F1, confusion matrices, and ROC 
   - Best performance from Random Forest (76.3% test accuracy)

----

## How to Run

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/moodify.git
cd moodify
pip install -r requirements.txt