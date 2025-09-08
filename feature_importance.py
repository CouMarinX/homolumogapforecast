import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


def plot_feature_importance():
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names

    model = RandomForestRegressor(random_state=0)
    model.fit(X, y)
    importances = model.feature_importances_

    # Order features by importance
    indices = importances.argsort()[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importance plot saved to feature_importance.png")


if __name__ == "__main__":
    plot_feature_importance()
