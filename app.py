from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json
import matplotlib.pyplot as plt
import openai
from datetime import datetime, timedelta

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Folder Configurations
UPLOAD_FOLDER = './static/uploads/'
REPORT_FOLDER = './static/reports/'
VISUAL_FOLDER = './static/visuals/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['VISUAL_FOLDER'] = VISUAL_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(VISUAL_FOLDER, exist_ok=True)

# OpenAI API Key
OPENAI_API_KEY = "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"  # Replace with your actual OpenAI API key
openai.api_key = OPENAI_API_KEY

# Constants
TARGET_CPA = 50
BUDGET_CHANGE_RATE = 0.2


# Debugging Helper
def debug_log(message):
    print(f"[DEBUG] {message}")


# Utility: Load and Preprocess Campaign Data
def load_and_preprocess(filepath):
    """Load and preprocess campaign data."""
    debug_log(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    data.rename(columns=lambda x: x.strip().lower(), inplace=True)

    # Ensure required columns exist
    required_columns = ['campaign id', 'impressions', 'clicks', 'conversions', 'spend', 'revenue', 'status']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Rename columns for uniformity
    data.rename(columns={
        'campaign id': 'Campaign ID',
        'impressions': 'Impressions',
        'clicks': 'Clicks',
        'conversions': 'Conversions',
        'spend': 'Spend',
        'revenue': 'Revenue',
        'status': 'Status'
    }, inplace=True)

    # Calculate metrics
    data['CTR'] = (data['Clicks'] / data['Impressions']) * 100
    data['CPA'] = data['Spend'] / data['Conversions'].replace(0, 1)
    data['ROAS'] = data['Revenue'] / data['Spend'].replace(0, 1)

    debug_log("Data successfully loaded and processed.")
    return data


# Utility: Optimize Campaigns
def optimize_campaigns(data, historical_data=None):
    """Apply optimization rules to campaign data."""
    actions = []
    today = datetime.now().date()

    for _, row in data.iterrows():
        action = {"Campaign ID": row['Campaign ID'], "Action": None, "Reason": None}

        # Pause Campaign
        if row['CTR'] < 1.0:
            action['Action'] = "Pause"
            action['Reason'] = f"Low CTR ({row['CTR']:.2f}%)"
        elif row['CPA'] > 3 * TARGET_CPA:
            action['Action'] = "Pause"
            action['Reason'] = f"High CPA (${row['CPA']:.2f})"

        # Increase Budget
        elif row['ROAS'] > 4:
            action['Action'] = "Increase Budget"
            action['Reason'] = f"High ROAS ({row['ROAS']:.2f})"
        elif historical_data is not None:
            # Week-over-week increase in conversions > 20%
            past_week_conversions = historical_data.loc[
                (historical_data['Campaign ID'] == row['Campaign ID']) &
                (historical_data['Date'] >= today - timedelta(days=7)),
                'Conversions'
            ].sum()
            current_week_conversions = row['Conversions']
            if past_week_conversions > 0 and (current_week_conversions - past_week_conversions) / past_week_conversions > 0.2:
                action['Action'] = "Increase Budget"
                action['Reason'] = "Conversions increased by more than 20% week-over-week."

        # Decrease Budget
        elif historical_data is not None:
            # ROAS < 1.5 for 5 consecutive days
            last_5_days_roas = historical_data.loc[
                (historical_data['Campaign ID'] == row['Campaign ID']) &
                (historical_data['Date'] >= today - timedelta(days=5)),
                'ROAS'
            ]
            if len(last_5_days_roas) == 5 and all(roas < 1.5 for roas in last_5_days_roas):
                action['Action'] = "Decrease Budget"
                action['Reason'] = "Low ROAS (< 1.5) for 5 consecutive days."

        if action['Action']:
            actions.append(action)

    debug_log(f"Generated actions: {actions}")
    return actions


# Utility: Generate Visualizations
def generate_visualizations(data):
    """Generate and save visualizations."""
    debug_log("Generating visualizations...")
    try:
        sns.barplot(x="Campaign ID", y="ROAS", hue="Status", data=data)
        roas_path = os.path.join(app.config['VISUAL_FOLDER'], 'roas_visualization.png')
        plt.title("ROAS by Campaign")
        plt.savefig(roas_path)
        plt.clf()
        debug_log(f"ROAS visualization saved at {roas_path}")

        sns.barplot(x="Campaign ID", y="CTR", hue="Status", data=data)
        ctr_path = os.path.join(app.config['VISUAL_FOLDER'], 'ctr_visualization.png')
        plt.title("CTR by Campaign")
        plt.savefig(ctr_path)
        plt.clf()
        debug_log(f"CTR visualization saved at {ctr_path}")
    except Exception as e:
        debug_log(f"Error generating visualizations: {e}")
        raise


# Utility: Generate LLM-Powered Insights
def generate_insights(data):
    """Generate insights using OpenAI GPT."""
    try:
        # Filter high and low performing campaigns for analysis
        high_performing = data[data['ROAS'] > 4]
        low_performing = data[data['ROAS'] < 1.5]

        # Prepare the prompt as a conversation
        prompt = (
            "Analyze the following marketing campaign data and provide optimization suggestions:\n\n"
            f"High Performing Campaigns: {high_performing.to_dict(orient='records')}\n\n"
            f"Low Performing Campaigns: {low_performing.to_dict(orient='records')}\n\n"
            "Suggestions:\n"
        )

        # Create the API call to the ChatCompletion endpoint
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in marketing campaign analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300  # Adjust max_tokens as needed for more content
        )

        # Return the generated insights
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        # Handle any exceptions
        debug_log(f"Error generating insights: {e}")
        return f"Error generating insights: {e}"




@app.route('/')
def index():
    """Homepage to upload files."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        flash("No file selected.")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        data = load_and_preprocess(filepath)

        # Placeholder for historical data (if available)
        historical_data = None

        actions = optimize_campaigns(data, historical_data)
        insights = generate_insights(data)
        generate_visualizations(data)

        # Save report
        report_path = os.path.join(REPORT_FOLDER, 'report.json')
        with open(report_path, 'w') as report_file:
            json.dump({"actions": actions, "insights": insights}, report_file)

        flash("Processing complete.")
        return redirect(url_for('insights'))
    except Exception as e:
        flash(f"Error processing file: {e}")
        return redirect(url_for('index'))


@app.route('/insights')
def insights():
    """Display the generated insights and actions."""
    report_path = os.path.join(REPORT_FOLDER, 'report.json')
    if not os.path.exists(report_path):
        flash("No report available.")
        return redirect(url_for('index'))

    with open(report_path, 'r') as file:
        report = json.load(file)

    return render_template('insights.html', report=report)


if __name__ == '__main__':
    app.run(debug=True)
