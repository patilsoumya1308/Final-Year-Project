
from app import app, db, generate_synthetic_data, train_model
import os

if __name__ == "__main__":
    # Generate data and train model if files don't exist
    if not os.path.exists('branch_prediction_data.csv'):
        generate_synthetic_data()
    if not os.path.exists('branch_predictor.pkl'):
        train_model()
        
    # Initialize database
    with app.app_context():
        db.create_all()
        
    # Start server
    app.run(host='0.0.0.0', port=5000)
