# Chance of Admission Prediction — System Architecture

```mermaid
graph TD
    A[👤 User] -->|Inputs details| B[🌐 Streamlit Dashboard / FastAPI]
    
    subgraph Data Pipeline
        B --> C[✅ Input Validation<br><i>(validator.py)</i>]
        C --> D[🔧 Preprocessing<br><i>(missing values, outliers)</i>]
        D --> E[⚙️ Feature Engineering<br><i>(encode, scale, split)</i>]
    end
    
    subgraph Machine Learning
        E --> F[📦 sklearn Pipeline<br><i>(ColumnTransformer)</i>]
        F --> G[🤖 Best Model<br><i>(XGBoost/LightGBM)</i>]
        G --> H[📊 Prediction]
    end
    
    subgraph Explainability & Logging
        H --> I[📝 Prediction History<br><i>(predictions.csv)</i>]
        H --> J[🔍 SHAP Explanation<br><i>(explain.py)</i>]
        J --> K[🎯 Result Display]
    end
    
    subgraph Experiment Tracking
        L[🧠 MLflow] -.->|Tracks| F
        L -.->|Tracks| G
    end

    classDef UI fill:#4CAF50,stroke:#2E7D32,color:white;
    classDef Pipeline fill:#2196F3,stroke:#1565C0,color:white;
    classDef ML fill:#9C27B0,stroke:#6A1B9A,color:white;
    classDef Logs fill:#FF9800,stroke:#EF6C00,color:white;

    class A,B,K UI;
    class C,D,E Pipeline;
    class F,G,H,L ML;
    class I,J Logs;
```

## Flow Description

1. **User Input:** A user inputs their academic and demographic profile either through the Streamlit web dashboard or the FastAPI REST endpoints.
2. **Validation Layer:** Inputs are strictly validated against rules defined in `validator.py` and `config.yaml`.
3. **Pipeline Transformation:** A saved scikit-learn `Pipeline` (with a `ColumnTransformer`) correctly scales numerical features and encodes categorical/ordinal features.
4. **Model Inference:** The best performing ensemble model predicts the probability of admission.
5. **Explainability:** SHAP values are calculated on-the-fly to explain *why* the prediction was made (e.g., "CGPA boosted chances by 18%").
6. **Logging:** The input and prediction are appended to `predictions.csv` for audit trails and monitoring.
