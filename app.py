import os
import pandas as pd
import random
from flask import Flask, request, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ========== Step 1: Generate Synthetic Dataset ==========
def generate_synthetic_data(num_rows=100000):
    categories = ['GEN', 'OBC', 'SC', 'ST']
    branches = ['CSE', 'IT', 'ECE', 'ME', 'CE']
    data = []

    for i in range(num_rows):
        jee = round(random.uniform(50, 100), 2) if random.random() > 0.5 else ''
        cet = round(random.uniform(50, 100), 2) if jee == '' else ''
        category = random.choice(categories)
        tenth = round(random.uniform(70, 100), 2)
        twelfth = round(random.uniform(70, 100), 2)
        branch = random.choice(branches)
        data.append([i + 1, jee, cet, category, tenth, twelfth, branch])

    df = pd.DataFrame(data, columns=[
        'id', 'jee_percentile', 'mhtcet_percentile', 'category',
        'tenth_percentage', 'twelfth_percentage', 'predicted_branch'
    ])
    df.to_csv('branch_prediction_data.csv', index=False)
    print("✅ CSV generated: branch_prediction_data.csv")

# ========== Step 2: Train Model ==========
def train_model():
    df = pd.read_csv('branch_prediction_data.csv')
    df['jee_percentile'].fillna(0, inplace=True)
    df['mhtcet_percentile'].fillna(0, inplace=True)

    le_category = LabelEncoder()
    le_branch = LabelEncoder()

    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['branch_encoded'] = le_branch.fit_transform(df['predicted_branch'])

    X = df[['jee_percentile', 'mhtcet_percentile', 'category_encoded', 'tenth_percentage', 'twelfth_percentage']]
    y = df['branch_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'branch_predictor.pkl')
    joblib.dump(le_category, 'category_encoder.pkl')
    joblib.dump(le_branch, 'branch_encoder.pkl')

    print("✅ Model and encoders saved: branch_predictor.pkl, category_encoder.pkl, branch_encoder.pkl")

# ========== Step 3: Flask Web Application ==========

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

def init_db():
    with app.app_context():
        db.create_all()
        
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=email).first():
            return "User already exists"
        hashed_password = generate_password_hash(password)
        user = User(username=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user'] = username
            return redirect('/predict')
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        try:
            exam_type = request.form.get('exam_type')
            exam_score = float(request.form.get('exam_score'))
            jee = exam_score if exam_type == 'jee' else 0
            cet = exam_score if exam_type == 'cet' else 0
            category = request.form['category']
            tenth = float(request.form['tenth'])
            twelfth = float(request.form['twelfth'])

            model = joblib.load('branch_predictor.pkl')
            le_category = joblib.load('category_encoder.pkl')
            le_branch = joblib.load('branch_encoder.pkl')

            cat_encoded = le_category.transform([category])[0]
            input_data = [[jee, cet, cat_encoded, tenth, twelfth]]
            pred = model.predict(input_data)
            branch = le_branch.inverse_transform(pred)[0]

            return render_template('result.html', 
                branch=branch,
                exam_type=exam_type,
                category=category,
                score=exam_score,
                tenth=tenth,
                twelfth=twelfth
            )
        except ValueError:
            return "Please enter valid numerical values for percentages", 400

    return render_template('predict.html')

# ========== Main Script ==========
if __name__ == '__main__':
    if not os.path.exists('branch_prediction_data.csv'):
        generate_synthetic_data()
    if not os.path.exists('branch_predictor.pkl'):
        train_model()
    with app.app_context():
        db.drop_all()  # Reset database
        db.create_all()  # Create fresh tables
    app.run(debug=True)