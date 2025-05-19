# Moodify

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

---

## Final Model

The best-performing model was a **fine-tuned Random Forest classifier**, trained to predict mood clusters based on Spotify audio features.

**Why Random Forest?**
- Handles mixed data types (numerical & categorical)
- Captures non-linear feature interactions
- Resistant to overfitting, especially with imbalanced data (via SMOTE)

**Best hyperparameters:**
- `n_estimators = 120`
- `max_depth = None`
- `min_samples_split = 2`
- `min_samples_leaf = 1`

---

## Training

The model was trained on a dataset of 232,000+ tracks, with the following steps:

- Unsupervised clustering on core audio features to define 5 pseudo-mood labels
- One-hot encoding of categorical features
- Standardized numerical features
- Added interaction terms (e.g., `speechiness * tempo`)
- Data split: **90% train**, **5% validation**, **5% test**
- Applied **SMOTE** to balance class distribution in training
- Hyperparameter tuning via **RandomizedSearchCV**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf.fit(X_train, y_train)
```

## Inference
Once trained, the model can predict the mood of a new song using its audio features. Use cases include:
- Auto-tagging music libraries by mood
- Generating mood-based playlists
- Recommending emotionally similar songs

```python
import pandas as pd

# Sample song features (preprocessed as needed)
song_features = pd.DataFrame([{
    'danceability': 0.78,
    'energy': 0.65,
    'loudness': -5.2,
    'speechiness': 0.04,
    'tempo': 120,
    'key_C': 0, 'key_D#': 1,  # one-hot encoded keys
    'time_signature_4/4': 1,
    # ... include all required engineered and aligned features
}])

# Predict mood
predicted_mood = rf.predict(song_features)
print("Predicted Mood:", predicted_mood)
```
Output:
One of the 5 moods: Calm, Content, Energetic, Happy, or Mellow

## How to Run

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/moodify.git
cd moodify
pip install -r requirements.txt
```
Then open the notebook:
```bash
jupyter notebook project.ipynb
```