import os
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from dbHandler import DBHandler
from helper import Helper

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from collections import Counter


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024

dbHandler = DBHandler()
helper = Helper()
dbHandler.setupEngine()

dataHandler = None


UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/healthy")
def health_check():
    return "healthy", 200

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({
        "error": "File too large",
        "message": "The uploaded file exceeds the allowed limit of 150 MB."
    }), 413

# Route for Basket Analysis data using Machine Learning model
@app.route("/get_basket_analysis_ml", methods=["GET"])
def get_basket_analysis_ml():
    # Load transaction data
    query = """
    SELECT basket_num, product_num
    FROM transactions
    LIMIT 10000
    """
    df = pd.read_sql(query, con=dbHandler.engine)

    # Group products by basket_num
    basket_groups = df.groupby("basket_num")["product_num"].apply(list)

    # Generate product pairs (combinations) for each basket
    product_pairs = Counter()
    for products in basket_groups:
        product_pairs.update(combinations(sorted(products), 2))

    # Convert to DataFrame for model input
    pairs_df = pd.DataFrame(
        [(pair[0], pair[1], count) for pair, count in product_pairs.items()],
        columns=["Product_A", "Product_B", "Frequency"]
    )

    # Encode Product_A and Product_B using LabelEncoder
    le = LabelEncoder()
    all_products = pd.concat([pairs_df['Product_A'], pairs_df['Product_B']]).unique()
    le.fit(all_products)
    pairs_df['Product_A_Encoded'] = le.transform(pairs_df['Product_A'])
    pairs_df['Product_B_Encoded'] = le.transform(pairs_df['Product_B'])

    # Feature engineering: we can use Frequency as the target variable
    X = pairs_df[['Product_A_Encoded', 'Product_B_Encoded']]
    y = pairs_df['Frequency']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train RandomForest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Model evaluation
    y_pred = rf_model.predict(X_test)
    model_report = classification_report(y_test, y_pred, output_dict=True)

    # Get top product pairs and their frequencies
    top_pairs = pairs_df.sort_values(by="Frequency", ascending=False).head(10)

    # Format model report for easier parsing in the front-end
    formatted_report = {
        "classification_report": model_report,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": model_report['accuracy'],
        "recall": model_report['macro avg']['recall'],
        "f1_score": model_report['macro avg']['f1-score'],
        "auc": None  # This can be calculated separately, if needed
    }

    # Redirect to basket_analysis.html and pass the data as part of the render
    return render_template("basket_analysis.html", top_pairs=top_pairs.to_dict(orient='records'), model_report=formatted_report)


# Route for the index page (User Registration)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")

        # Display the welcome message briefly and redirect to index
        return redirect(url_for('index', username=username, email=email))

    # If a GET request or after redirection, pass the username and email for rendering
    username = request.args.get('username')
    email = request.args.get('email')

    return render_template("index.html", username=username, email=email)

# Route for the data upload page
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        transactions_file = request.files.get("transactions_file")
        households_file = request.files.get("households_file")
        products_file = request.files.get("products_file")

        if households_file and "households" in households_file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "households")
            households_file.save(filepath)
            helper.cleanAndUpload(filepath, "households")

        if transactions_file and "transactions" in transactions_file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "transactions")
            transactions_file.save(filepath)
            helper.cleanAndUpload(filepath, "transactions")

        if products_file and "products" in products_file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "products")
            products_file.save(filepath)
            helper.cleanAndUpload(filepath, "products")

        return redirect(url_for("upload", message="Files uploaded successfully!"))

    return render_template("upload.html")

# Route for the search page
@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        results = helper.getRetailDataforHH(request.form.get("hshd_num"))
    
    return render_template("search.html", results=results)

# Route for displaying data for HSHD_NUM 10
@app.route("/samplepull", methods=["GET"])
def sample_pull():
    # Query to fetch data for HSHD_NUM = 10
    hshd_num = 10
    results = helper.getRetailDataforHH(hshd_num)

    # Render the results on samplepull.html
    return render_template("samplepull.html", results=results)

# Route for the dashboard page
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Route for Demographics and Engagement data
@app.route("/get_demographics_data", methods=["GET"])
def get_demographics_data():
    query = """
    SELECT hh_size, children, COUNT(*) as frequency
    FROM households
    GROUP BY hh_size, children
    """
    df = pd.read_sql(query, con=dbHandler.engine)
    return jsonify(df.to_dict(orient='records'))

# Route for Engagement Over Time data
@app.route("/get_engagement_over_time", methods=["GET"])
def get_engagement_over_time():
    query = """
    SELECT year, SUM(spend) AS total_spend
    FROM transactions
    GROUP BY year
    ORDER BY year
    """
    df = pd.read_sql(query, con=dbHandler.engine)
    return jsonify(df.to_dict(orient='records'))

# Route for Basket Analysis data
@app.route("/get_basket_analysis", methods=["GET"])
def get_basket_analysis():

    query = """
    SELECT itemset_label, frequency
    FROM frequent_itemsets_results
    ORDER BY frequency DESC
    """
    frequent_itemsets = pd.read_sql(query, con=dbHandler.engine)

    return jsonify(frequent_itemsets.to_dict(orient='records'))


# Route for Seasonal Trends data
@app.route("/get_seasonal_trends", methods=["GET"])
def get_seasonal_trends():
    query = """
    SELECT week_num, SUM(spend) AS weekly_spend
    FROM merged_retail_data
    GROUP BY week_num
    ORDER BY week_num
    """
    df = pd.read_sql(query, con=dbHandler.engine)
    return jsonify(df.to_dict(orient='records'))

# Route for Brand Preferences data
@app.route("/get_brand_preferences", methods=["GET"])
def get_brand_preferences():
    query = """
    SELECT brand_type, COUNT(*) AS frequency
    FROM products
    GROUP BY brand_type
    ORDER BY frequency DESC
    """
    df = pd.read_sql(query, con=dbHandler.engine)
    return jsonify(df.to_dict(orient='records'))

# Route for Churn Prediction data
@app.route('/get_churn_predictions', methods=['GET'])
def get_churn_predictions():
    query = """
    SELECT hshd_num, loyalty_flag, age_range, 
           hh_size, children
    FROM households
    """
    df = pd.read_sql(query, con=dbHandler.engine)

    df.dropna(axis=0, subset=["hh_size", "children"])
    # Convert 'HH_size' and 'Children' columns to numeric (int)
    df['hh_size'] = pd.to_numeric(df['hh_size'], errors='coerce')  # Handle non-numeric values gracefully
    df['children'] = pd.to_numeric(df['children'], errors='coerce')

    # Churn Prediction Logic
    def calculate_churn_risk(row):
        if row['loyalty_flag'] == 'N' or (row['hh_size'] <= 2 and row['children'] == 0):
            return 'High'
        return 'Low'

    df['churn_risk'] = df.apply(calculate_churn_risk, axis=1)

    # Aggregate data for graphical representation
    churn_by_age = df.groupby('age_range')['churn_risk'].apply(lambda x: (x == 'High').sum()).reset_index()
    churn_by_age.columns = ['age_range', 'high_churn_count']

    churn_distribution = df['churn_risk'].value_counts().reset_index()
    churn_distribution.columns = ['churn_risk', 'count']

    return jsonify({
        'churn_by_age': churn_by_age.to_dict(orient='records'),
        'churn_distribution': churn_distribution.to_dict(orient='records')
    })


if __name__ == "__main__":
    app.run()
