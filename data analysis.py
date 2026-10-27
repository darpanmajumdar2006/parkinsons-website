from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
DATASET_PATH = Path(r"C:\Users\DARPAN\Downloads\parkinsons2 (2)\pd_speech_features\pd_dataset.csv")
def main() -> None:
    df = pd.read_csv(DATASET_PATH, header=1)
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nNon-null counts:")
    print(df.count())
    print("\nStatistical summary:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isna().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)

    target_candidates = [column for column in df.columns if any(word in column.lower() for word in ("id", "class", "status", "label", "target", "diagnosis"))]
    print("\nPotential target or identifier columns:")
    print(target_candidates)

    rows_before_dropna = len(df)
    df.dropna(inplace=True)
    print(f"\nRows removed due to missing values: {rows_before_dropna - len(df)}")
    print(f"Shape after removing missing values: {df.shape}")
    print(f"Duplicate rows after removing missing values: {df.duplicated().sum()}")
    print("\nClass distribution:")
    class_counts = df["class"].value_counts()
    print(class_counts)
    
    
    plt.bar(class_counts.index.astype(str), class_counts.values)
    plt.title("Distribution of Parkinson's classes")
    plt.xlabel("class")
    plt.ylabel("number of samples")
    plt.tight_layout()
    plt.show()
    print(df['gender'].value_counts())
    counts = df.groupby(['class', 'gender']).size()
    counts = counts.unstack()
    print(counts)
    counts.plot(kind="bar")
    plt.title("Gender distribution by Parkinson's Disease Class")
    plt.xlabel("class")
    plt.ylabel("Number of samples")
    plt.legend(title="gender")
    plt.tight_layout()
    plt.xticks(rotation=0) #rotation of x axis
    plt.show()

    sns.histplot(data=df, x='PPE', hue='class', kde=True)
    plt.title('Distribution of PPE by Parkinson’s Disease Class')
    plt.xlabel('PPE (Pitch Period Entropy)')
    plt.ylabel('Frequency')
    plt.xlim(0.65, 0.9)
    plt.tight_layout()
    plt.figure(figsize=(10, 6))
    plt.show()

    sns.histplot(data=df, x='DFA', hue='class', kde=True)
    plt.title('Distribution of DFA by Parkinson’s Disease Class')
    plt.xlabel('DFA (Detrend Fluctuation Analysis)')
    plt.ylabel('Frequency')
    plt.show()
    
    sns.histplot(data=df, x='RPDE', hue='class', kde=True)
    plt.title('Distribution of RPDE by Parkinson’s Disease Class')
    plt.xlabel('RPDE (Recurrence Period Density Entropy)')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="meanPeriodPulses", hue="class", bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of Mean Pulse Period by Parkinson's Disease Class")
    plt.xlabel("Mean pulse period (seconds)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='locPctJitter', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of Local Jitter by Parkinson's Disease Class")
    plt.xlabel("Local Percentage Jitter")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='locShimmer', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of Local Shimmer by Parkinson's Disease Class")
    plt.xlabel("Local Shimmer")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='locDbShimmer', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of Local DB Shimmer by Parkinson's Disease Class")
    plt.xlabel("Local Db Shimmer")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='apq3Shimmer', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of APQ3 Shimmer by Parkinson's Disease Class")
    plt.xlabel("APQ3 Shimmer")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='numPulses', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of Number of Pulses by Parkinson's Disease Class")
    plt.xlabel("Number of Pulses")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='stdDevPeriodPulses', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of Pulse Period Standard Deviation by Parkinson's Disease Class")
    plt.xlabel("Standard Deviation of Pulse Periods")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='apq5Shimmer', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of APQ5 Shimmer by Parkinson's Disease Class")
    plt.xlabel("APQ5 Shimmer")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='apq11Shimmer', hue='class', bins=20, kde=True, alpha=0.5)
    plt.title("Distribution of APQ11 Shimmer by Parkinson's Disease Class")
    plt.xlabel("APQ11 Shimmer")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("\nClass distribution:")
    print(df['class'].value_counts())
    print("\nTotal missing values:", df.isnull().sum().sum())
    print("\nPossible ID columns:")
    print([col for col in df.columns if 'id' in str(col).lower() or 'subject' in str(col).lower()])

    from sklearn.model_selection import train_test_split
    x=df.drop(columns=['class']) #separating features and target variable
    y=df['class']
    id_cols=[col for col in x.columns if col.lower() in ['id','subject_id', 'subjectid','subject']] #remove identifier columns if confirmed
    x=x.drop(columns=id_cols)
    print("feature matrix shape:", x.shape)
    print("Target Shape:",y.shape)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
    print("Training set shape:", x_train.shape)
    print("Test set shape:", x_test.shape)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,ConfusionMatrixDisplay,classification_report)
    models={"Logistic Regression":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))]), #Logistic Regression with Standard Scaler
            "SVM":Pipeline([("scaler",StandardScaler()),("model",SVC(probability=True,class_weight="balanced",random_state=42))]), #SVM with Standard Scaler
            "Random Forest":Pipeline([("scaler",StandardScaler()),("model",RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1))])} #Random Forest with Standard Scaler
    results=[]
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for (name,model), ax in zip(models.items(),axes):
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        if hasattr(model,"predict_proba"): #get probability score for roc-auc
            y_score=model.predict_proba(x_test)[:, 1]
        else:
            y_score=model.decision_function(x_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_score),
        })
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["Class 0", "Class 1"],
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )
        ax.set_title(f"{name}\nConfusion Matrix")
    plt.tight_layout()
    plt.show()
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.round(4).to_string(index=False))

    print(f"\n{name} classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    pca_models = {
    "Logistic Regression + PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42))]),
    "SVM + PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("model", SVC(
            probability=True, class_weight="balanced", random_state=42))])}

    pca_results = []

    for name, model in pca_models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]

        pca_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_score),
            "PCA Components": model.named_steps["pca"].n_components_
        })
    print("\nPCA model comparison:")
    print(pd.DataFrame(pca_results).round(4).to_string(index=False))

    rf_model = models["Random Forest"]
    importance_df = pd.DataFrame({
        "Feature": x_train.columns,
        "Importance": rf_model.named_steps["model"].feature_importances_,
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    print("\nTop 15 Important Voice Features:")
    print(importance_df.head(15).to_string(index=False))
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15),x="Importance",y="Feature")
    plt.title("Top 15 Features for Parkinson's Classification")
    plt.xlabel("Feature Importance")
    plt.ylabel("Voice Feature")
    plt.tight_layout()
    plt.show()




    

if __name__ == "__main__":
    main()
