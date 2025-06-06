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

Once trained, the model can predict the mood of a new song using its audio features.  
This can be useful for:
- Auto-tagging music libraries by mood
- Creating mood-based playlists or filters
- Recommending emotionally similar songs

We’ve included a built-in function at the end of the notebook that allows you to automatically receive a predicted mood and manual one where you can input audio features and receive a predicted mood.

### Usage Example
#### Integration with Spotify API

You can connect Moodify to the **Spotify Web API** to automatically fetch the audio features of any song using its Spotify track ID or name.

#### Get Audio Features from Spotify

To use the Spotify API, follow these steps:

1. **Create a Spotify Developer Account**  
   - Go to [developer.spotify.com](https://developer.spotify.com/)
    ⚠️ Note: as of 2024–2025, many users (especially student apps or public IPs) have reported that audio-features returns 403s unless your app is:
	- Associated with an approved Spotify Partner
	- Or has been explicitly authorized by Spotify’s internal team
   - Log in and create a new app
   - Copy your **Client ID** and **Client Secret**

2. **Install Spotipy**

```bash
pip install spotipy
```

After training and running all cells, simply run:

```python
# fill in your Spotify credentials
client_id = "YOUR_SPOTIFY_CLIENT_ID"
client_secret = "YOUR_SPOTIFY_CLIENT_SECRET"

# get the song features
df = get_features_from_spotify("Blinding Lights", client_id, client_secret)

# Predict mood using your model
if df is not None:
    predict_mood_from_df() 
```

And if you would like to input the features manually you can run this:
```python
predict_mood_manual_input()
```

⚠️ Note: The `predict_mood_manual_input()` function requires user input via the terminal.
If you are running this in a Jupyter Notebook (especially in a browser or cloud environment), manual typing may not be supported.

To use this function, we recommend exporting the notebook as a `.py` script and running it from a terminal:

```bash
jupyter nbconvert --to script project.ipynb
python project.py
```


You will be prompted to enter values for features like danceability, energy, loudness, and encoded categories like key_C or time_signature_4/4. The model will return:
- The predicted mood cluster
- The probabilities for all classes

Example Output:
- Predicted Mood: Content
- Class Probabilities:
 - - Calm: 0.03
 - - Content: 0.76
 - - Energetic: 0.10
 - - Happy: 0.06
 - - Mellow: 0.05
 - A visual

## Acknowledgements
- Audio features from Spotify Web API
- Built with pandas, scikit-learn, xgboost, seaborn, and imbalanced-learn
- Developed as part of a university machine learning project

