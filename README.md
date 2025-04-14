# ⚡ Energy Consumption Prediction using XGBoost

Welcome to the **Energy Consumption Prediction** MLOps project! This repository demonstrates the end-to-end development and deployment of a machine learning pipeline using **XGBoost** to predict energy consumption. The pipeline is designed using modern MLOps practices with CI/CD integration, cloud storage, and robust validation.

## 🧱 Project Structure

This project is split into two main pipelines:

### 🚛 Training Pipeline
1. **Data Ingestion**
   - Fetches raw data from MongoDB.  
   - Saves the data in the `artifacts/` directory.

2. **Data Validation**  
   - Validates the schema against `schema.yaml`.  
   - Records validation status and stores valid data in `artifacts/`.

3. **Data Transformation**  
   - Applies transformations (MinMax or Standard Scaler as defined in schema).  
   - Stores the transformed data in `artifacts/`.

4. **Model Trainer**  
   - Trains the XGBoost model.  
   - Saves the trained model to `artifacts/`.

5. **Model Evaluation**  
   - Evaluates the trained model on test data.  
   - Computes accuracy and performance metrics.

6. **Model Pusher**  
   - Pulls the previous best model from **AWS S3**.  
   - Compares the new model; if the accuracy is at least **0.02%** better, pushes it to the S3 bucket.

### 🔮 Prediction Pipeline
  - The best performing model is deployed to an **AWS EC2 (t2.micro)** instance in **us-east-1**.
  - A FastAPI is exposed for inference via a lightweight web application.
  - CI/CD pipeline is implemented using **GitHub Actions** for streamlined deployment and updates.

## 🛠️ Tech Stack

  - **Machine Learning**: XGBoost, Scikit-learn  
  - **MLOps**: Docker, GitHub Actions, CI/CD  
  - **Cloud**: AWS EC2, AWS S3, MongoDB Atlas  
  - **Web Framework**: FastAPI / Flask (Specify if used)  
  - **Python Version**: 3.10

## 🚀 Running Locally

### Clone this model locally
```bash
git clone https://github.com/PrathamDesai07/Energy-Consumption-Prediction
```

### 🔐 Environment Variables

Before running the project, create a `.env` file or export the following variables:

```bash
export MONGODB_URL="your_mongodb_url"
export AWS_ACCESS_KEY_ID="your_s3_bucket_url"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key"
export AWS_DEFAULT_REGION="your_aws_secret_access_key"
```

> 💡 Tip: On Windows PowerShell, use `$env:VAR_NAME="value"` format.

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 📦 Train Model Locally

```bash
python demo.py
```

### 📦 To perform evaluation and run the stored model

```bash
python app.py
```

## 🔄 CI/CD Pipeline

This project includes a CI/CD pipeline using **GitHub Actions**:

  - Pushes code to EC2 instance automatically on changes.
  - Triggers retraining or re-deployment if necessary.
  - Ensures reproducibility and robustness in development workflows.

## 📁 Artifacts Directory

The `artifacts/` folder stores all intermediate and final outputs such as:

  - Raw / validated / transformed data
  - Trained model binaries
  - Evaluation reports
  - Logs

## 📊 Performance Monitoring

Model evaluation is performed on:

  - MSE/MAE
  - Improvement tracking for auto-push to S3

## 📬 Connect & Contribute

Have questions or want to contribute?  
Feel free to [open an issue](https://github.com/yourusername/energy-xgboost/issues) or submit a PR!

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🌟 Acknowledgments

- MongoDB Atlas for cloud database  
- AWS for model storage and deployment  
- GitHub Actions for seamless CI/CD  
- XGBoost for reliable performance in prediction
