from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.feature_selection import mutual_info_classif as cuml_mutual_info
import cudf
import pandas as pd
import json

# ----------------------------
# Load dataset into cuDF (GPU dataframe)
# ----------------------------
nba_dataset = pd.read_csv("nba_dataset.csv", sep="\t", encoding="utf-8")
nba_dataset['is_top_75'] = nba_dataset["full_name"].apply(
    lambda name: 1 if name in [
        "Kareem Abdul-Jabbar", "Ray Allen", "Giannis Antetokounmpo", "Carmelo Anthony",
        "Nate Archibald", "Paul Arizin", "Charles Barkley", "Rick Barry", "Elgin Baylor",
        "Dave Bing", "Larry Bird", "Kobe Bryant", "Wilt Chamberlain", "Bob Cousy",
        "Dave Cowens", "Billy Cunningham", "Stephen Curry", "Anthony Davis", "Dave DeBusschere",
        "Clyde Drexler", "Tim Duncan", "Kevin Durant", "Julius Erving", "Patrick Ewing",
        "Walt Frazier", "Kevin Garnett", "George Gervin", "Hal Greer", "James Harden",
        "John Havlicek", "Elvin Hayes", "Allen Iverson", "LeBron James", "Magic Johnson",
        "Sam Jones", "Michael Jordan", "Jason Kidd", "Kawhi Leonard", "Damian Lillard",
        "Jerry Lucas", "Karl Malone", "Moses Malone", "Pete Maravich", "Bob McAdoo",
        "Kevin McHale", "George Mikan", "Reggie Miller", "Earl Monroe", "Steve Nash",
        "Dirk Nowitzki", "Shaquille O'Neal", "Hakeem Olajuwon", "Robert Parish", "Chris Paul",
        "Gary Payton", "Bob Pettit", "Paul Pierce", "Scottie Pippen", "Willis Reed",
        "Oscar Robertson", "David Robinson", "Dennis Rodman", "Bill Russell", "Dolph Schayes",
        "Bill Sharman", "John Stockton", "Isiah Thomas", "Nate Thurmond", "Wes Unseld",
        "Dwyane Wade", "Bill Walton", "Jerry West", "Russell Westbrook", "Lenny Wilkens",
        "Dominique Wilkins", "James Worthy"
    ] else 0
)

# Convert pandas dataframe to cuDF
nba_cudf = cudf.DataFrame.from_pandas(nba_dataset)

# Save filtered dataset (optional)
nba_cudf.to_pandas().to_csv("nba_filtered_dataset.txt", sep="\t", index=False, encoding="utf-8")

# ----------------------------
# Features
# ----------------------------
feature_columns = [
    "NBA All-Star", "MVP", "All-NBA-First", "All-NBA-Second", "All-NBA-Third",
    "DPOY", "All-Defensive-First", "All-Defensive-Second", "Championships",
    "FMVP", "Olympic Gold", "Olympic Silver", "Olympic Bronze", 'season_PPG',
    'season_APG', 'season_RBG', 'season_SPG', 'season_BPG', 'season_TPG',
    'season_MPG', 'season_FG_PCT', 'season_FG3_PCT', 'season_FT_PCT', 'season_PM',
    'playoff_PPG', 'playoff_APG', 'playoff_RBG', 'playoff_SPG', 'playoff_BPG',
    'playoff_TPG', 'playoff_MPG', 'playoff_FG_PCT', 'playoff_FG3_PCT',
    'playoff_FT_PCT', 'playoff_PM'
]

# ----------------------------
# Main function
# ----------------------------
def main():
    X = nba_cudf[feature_columns]
    y = nba_cudf['is_top_75']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ----------------------------
    # RandomForestClassifier on GPU
    # ----------------------------
    clf = cuRF(
        n_estimators=500,
        max_depth=6,
        max_features=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).sum() / y_test.shape[0]

    # ----------------------------
    # Feature selection: mutual information (GPU)
    # ----------------------------
    mi_scores = cuml_mutual_info(X, y)
    mi_dict = dict(zip(feature_columns, mi_scores))

    # ----------------------------
    # Correlations (still CPU, cuDF->pandas)
    # ----------------------------
    correlations = nba_cudf.to_pandas().corr(numeric_only=True)["is_top_75"].drop("is_top_75").sort_values(key=abs, ascending=False)

    # ----------------------------
    # Predict Top 75
    # ----------------------------
    probs = clf.predict_proba(X)[:, 1]
    nba_cudf['predicted_prob'] = probs
    top_75_predicted = nba_cudf.sort_values(by='predicted_prob', ascending=False).head(75)
    result = top_75_predicted['full_name'].to_pandas().tolist()

    return result

# ----------------------------
# Run script safely
# ----------------------------
if __name__ == "__main__":
    try:
        top75 = main()
        if not top75:
            top75 = []
        print(json.dumps(top75, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))