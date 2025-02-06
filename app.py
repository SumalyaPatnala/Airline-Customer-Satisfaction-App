import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearnmetrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
import joblib
import scipy.stats as stats


# Load and preprocess data
def load_data():
    df = pd.read_csv("Airline.csv")

    # Encoding categorical variables
    satisfaction_mapping = {"satisfied": 1, "dissatisfied": 0}
    df['satisfaction'] = df['satisfaction'].map(satisfaction_mapping)

    gender_mapping = {"Female": 1, "Male": 0}
    df['Gender'] = df['Gender'].map(gender_mapping)

    travel_type_mapping = {"Business travel": 2, "Personal Travel": 1}
    df['Type of Travel'] = df['Type of Travel'].map(travel_type_mapping)

    class_mapping = {"Business": 3, "Eco Plus": 2, "Eco": 1}
    df['Class'] = df['Class'].map(class_mapping)

    # Filter the dataset to include only the specified columns
    columns_to_keep = ['Gender', 'Age', 'Type of Travel', 'Class', 'Seat comfort', 'Online support',
                       'Inflight wifi service', 'Departure Delay in Minutes', 'Cleanliness', 'satisfaction']
    df = df[columns_to_keep]

    return df


# Split data
def split_data(df):
    target = 'satisfaction'
    X = df.drop([target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                      random_state=42)

    return X_train, X_test, y_train, y_test


# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Save model
def save_model(model):
    joblib.dump(model, 'random_forest_model.pkl')


# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Methodology", "Statistical Analysis", "Visualization", "Prediction",
                                          "Model Evaluation"])

    if page == "Home":
        show_home()
    elif page == "Methodology":
        show_methodology()
    elif page == "Statistical Analysis":
        show_statistical_analysis()
    elif page == "Visualization":
        show_visualization()
    elif page == "Prediction":
        show_prediction()
    elif page == "Model Evaluation":
        show_model_evaluation()


def show_home():
    st.title("Airline Customer Satisfaction APP")
    st.image("Picture1.jpg")
    st.write(
        "Welcome to the Airline Customer Satisfaction Analysis app. Use the navigation menu to explore different sections of the analysis.")


def show_methodology():
    st.title("Flow Diagram")
    flowchart = '''
        digraph {
            node [shape=rect, style=filled, fillcolor=lightblue, fontname=Arial, fontsize=12]
            Start [shape=ellipse, style=filled, fillcolor="#ffcc99"]
            LoadData [label="Load Data", fillcolor="#ffcc99"]
            Encode [label="Encode Categorical Variables", fillcolor="#ccffcc"]
            SplitData [label="Split Data", fillcolor="#ccffcc"]
            TrainModel [label="Train Random Forest Model", fillcolor="#ffccff"]
            EvaluateModel [label="Evaluate Model", fillcolor="#ffff99"]
            SaveModel [label="Save Model", fillcolor="#ccffff"]
            End [shape=ellipse, style=filled, fillcolor="#ffcc99"]

            Start -> LoadData
            LoadData -> Encode
            Encode -> SplitData
            SplitData -> TrainModel
            TrainModel -> EvaluateModel
            EvaluateModel -> SaveModel
            SaveModel -> End
        }
        '''
    # Render the flowchart
    st.graphviz_chart(flowchart)

    st.title("Conceptual Model")
    st.image("conceptual_model_airline.png")


def show_statistical_analysis():
    st.title("Statistical Analysis")
    df = load_data()
    st.write("Descriptive statistics of the dataset:")
    st.write(df.describe())
    st.write("Performing ANOVA...")
    anova_results = {}
    y = df['satisfaction']

    for column in ['Gender', 'Age', 'Type of Travel', 'Class', 'Seat comfort', 'Online support',
                   'Inflight wifi service', 'Departure Delay in Minutes', 'Cleanliness']:
        unique_values = df[column].unique()
        groups = [df[df[column] == val]['satisfaction'] for val in unique_values]
        anova_results[column] = stats.f_oneway(*groups)

    f_values = [result.statistic for result in anova_results.values()]
    p_values = [result.pvalue for result in anova_results.values()]

    anova_df = pd.DataFrame({
        'Feature': anova_results.keys(),
        'F-value': f_values,
        'p-value': p_values
    }).sort_values(by='p-value')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='F-value', y='Feature', data=anova_df, palette='viridis', ax=ax)
    ax.set_title("ANOVA F-values for Features")
    st.pyplot(fig)

    st.write("Interpreting the ANOVA results:")
    st.write("Features with p-value < 0.05 are considered statistically significant.")

    for index, row in anova_df.iterrows():
        if row['p-value'] < 0.05:
            st.success(f"{row['Feature']}: F-value = {row['F-value']:.2f}, p-value = {row['p-value']:.5f}")
        else:
            st.warning(f"{row['Feature']}: F-value = {row['F-value']:.2f}, p-value = {row['p-value']:.5f}")


def show_visualization():
    st.title("Visualization")
    df = load_data()
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            st.write(f"Distribution of {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)


def show_prediction():
    st.title("Prediction")
    st.write("Enter input data for prediction:")

    Gender = st.selectbox("Gender", [0, 1])
    Age = st.number_input("Age", 0, 100, 25)
    Type_of_Travel = st.selectbox("Type of Travel", [1, 2])
    Class = st.selectbox("Class", [1, 2, 3])
    Seat_comfort = st.slider("Seat comfort", 0, 5, 3)
    Online_support = st.slider("Online support", 0, 5, 3)
    Inflight_wifi_service = st.slider("Inflight wifi service", 0, 5, 3)
    Departure_Delay_in_Minutes = st.number_input("Departure Delay in Minutes", 0, 1000, 0)
    Cleanliness = st.slider("Cleanliness", 0, 5, 3)

    user_input = np.array([[Gender, Age, Type_of_Travel, Class, Seat_comfort, Online_support, Inflight_wifi_service,
                            Departure_Delay_in_Minutes, Cleanliness]])

    # if st.button("Predict"):
    #     df = load_data()
    #     X_train, X_test, y_train, y_test = split_data(df)
    #     model = train_model(X_train, y_train)
    #     save_model(model)
    #
    #     prediction = model.predict(user_input)
    #     st.write("Prediction (1: Satisfied, 0: Dissatisfied):", prediction[0])

    if st.button("Predict"):
        df = load_data()
        input_data = pd.DataFrame({
            'Gender': [Gender],
            'Age': [Age],
            'Type of Travel': [Type_of_Travel],
            'Class': [Class],
            'Seat comfort': [Seat_comfort],
            'Online support': [Online_support],
            'Inflight wifi service': [Inflight_wifi_service],
            'Departure Delay in Minutes': [Departure_Delay_in_Minutes],
            'Cleanliness': [Cleanliness]
        })

        X_train, X_test, y_train, y_test = split_data(df)
        model = train_model(X_train, y_train)
        save_model(model)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        probability_percentage = probability[0][1] * 100

        if probability_percentage > 50:  # Check if probability of satisfaction is greater than 50%
            st.success(
                f"Based on the input provided, the probability of the customer being satisfied is: {probability_percentage:.2f}%")
        else:
            st.warning(
                "Based on the input provided, the model predicts that the customer is likely to be dissatisfied.")


def show_model_evaluation():
    st.title("Model Evaluation")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("\nClassification Report:")
    st.write(classification_report(y_test, y_pred))

    st.write("\nAccuracy Score:")
    st.write(accuracy_score(y_test, y_pred))

    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    st.write("Feature Importance:")
    st.write(feature_importance_df)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    st.pyplot(fig)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importance (test set)")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
