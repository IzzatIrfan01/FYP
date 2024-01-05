import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import sklearn

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# StreamHandler to output log messages to Streamlit
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# Function to make predictions using the loaded model
def make_predictions(model, X):
    y_pred, sigma = model.predict(X, return_std=True)
    return y_pred, sigma

# Function to plot the results
def plot_results(filtered_data, category, predicted_waste_train, predicted_waste_test, date_test, date_train, sigma_range_test):
    # Visualization with Plotly
    # Create an interactive plot using Plotly
    fig = go.Figure()

    # Actual waste trace
    fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data[category], mode='markers', name='Actual Waste', marker=dict(color='#200fd6')))

    # Predicted Waste trace
    fig.add_trace(go.Scatter(x=date_train, y=predicted_waste_train, mode='lines', name='Predicted Waste (Train)', marker=dict(color='#f52222')))
    fig.add_trace(go.Scatter(x=date_test, y=predicted_waste_test, mode='lines', name='Predicted Waste (Test)', marker=dict(color='#00a110')))

    # Shaded uncertainty area
    fig.add_trace(go.Scatter(
        x=np.concatenate([date_test, date_test[::-1]]),
        y=np.concatenate([predicted_waste_test - 4 * sigma_range_test, (predicted_waste_test + 4 * sigma_range_test)[::-1]]),
        fill='toself',
        fillcolor='rgba(231,76,60,0.2)',  # Change the color here
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty'))

    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title=selected_category,
        title=f'Prediction waste for {selected_category}',
        showlegend=True,)

    st.plotly_chart(fig)

# Function to display item list table
def display_item_list_table(csv_path, search_input):
    try:
        item_df = pd.read_csv(csv_path)
    
        # Filter items based on search input
        if search_input:
            item_df = item_df[item_df['Item'].str.contains(search_input, case=False)]

        st.dataframe(item_df, hide_index=True)

    except FileNotFoundError:
        st.error("Item list CSV file not found. Please make sure the file exists.")

# Function for prediction food waste
def main(category):
    st.title("NutriMatch: ")

    # Read data
    df = pd.read_csv('src/Weekly_Average_FoodWaste.csv')

    # Convert date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define the chosen category for plotting
    chosen_category = category

    # Filter the data for the chosen category
    selected_columns = ['Date', chosen_category]
    filtered_data = df[selected_columns]
    # logger.info(filtered_data)

    split_ratio = 0.97
    split_index = int(len(filtered_data) * split_ratio)
    df_train = filtered_data[:split_index + 1]
    df_test = filtered_data[split_index:]

    # Extract features and target for the current state
    start = filtered_data['Date'].min()
    end = filtered_data['Date'].max()
    range_datetime = (end - start).days

    # Normalize date and waste variables
    reference_date = datetime(2023, 1, 1)
    normalized_date = (df_train['Date'] - reference_date).dt.days.values.reshape(-1, 1) / range_datetime
    normalized_waste = df_train[selected_category].values.reshape(-1, 1) / np.max(filtered_data[selected_category])

    X = normalized_date
    y = normalized_waste

    # Load the saved GP model
    model_filename = f"savedModel/{category}/{category} model.pkl"
    loaded_model = None

    # Check if the model file exists
    if st.button("Load Model"):
        try:
            loaded_model = joblib.load(model_filename)
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            # Handle the FileNotFoundError
            st.info(f" The food waste category of {category} does not has a saved model. Please choose another category.")
            return

    # Dummy figure
    dummy_fig = go.Figure()
    dummy_date = pd.to_datetime('2023-01-01')
    dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='markers', name='Actual Price'))
    dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='lines', name='Predicted Price',
                                line=dict(color='#e74c3c')))
    dummy_fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        title='Select a category to view the graph',
        showlegend=True,
    )

    # Make predictions when the model is loaded
    if loaded_model is not None:
        # prediction for train data
        start_date = df_train['Date'].min()
        end_date =   df_train['Date'].max() 
        date_train = pd.date_range(start=start_date, end=end_date, freq='D')

        # Normalize the date range
        normalized_date_train = (date_train - reference_date).days / range_datetime
        X_train = normalized_date_train.values.reshape(-1, 1)

        # Make predictions for the date range using the GP model
        y_pred_train, sigma_range_train = loaded_model.predict(X_train, return_std=True)

        # Denormalize the predicted wastes
        predicted_waste_train = y_pred_train * np.max(filtered_data[chosen_category])

        # predict for test data
        start_dates = df_test['Date'].min()
        end_dates =   df_test['Date'].max()+ timedelta(days=30)
        date_test = pd.date_range(start=start_dates, end=end_dates, freq='D')

        # Normalize the date range
        normalized_date_test = (date_test - reference_date).days / range_datetime
        X_test = normalized_date_test.values.reshape(-1, 1)

        # Make predictions for the date range using the GP model
        y_pred_test, sigma_range_test = loaded_model.predict(X_test, return_std=True)

        # Denormalize the predicted wastes
        predicted_waste_test = y_pred_test * np.max(filtered_data[chosen_category])

        # Plot results
        plot_results(filtered_data, category, predicted_waste_train, predicted_waste_test, date_test, date_train, sigma_range_test)
    else:
        # Display the dummy figure if the model is not loaded
        st.plotly_chart(dummy_fig)

# horizontal menu
selected = option_menu(
    menu_title= "NutriMatch", #required
    options=["Home","Prediction","Items"],
    icons=["house","bar-chart-line","file-earmark-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Add navigation bar with buttons
if selected == "Home":
    st.title(f"You have selected {selected}")

if selected == "Prediction":
    # Define the chosen category for plotting
    st.title(f"{selected} Visualization")
    selected_category = st.selectbox("Select Food Waste Category", ['Carbohydrates', 'Protein', 'Fat', 'Fiber'])
    main(selected_category)

if selected == "Items":
    st.title("Item List")
    # Sidebar for search input
    search_input = st.sidebar.text_input('Search Items:', '')
    # Specify the path to your item list CSV file
    item_list = "src\Merged_ItemList.csv"
    display_item_list_table(item_list,search_input)
    

    
