# Chance of Admission Prediction - System Architecture

```mermaid
graph TD
    A[User] -->|Inputs details| B[Streamlit Dashboard / FastAPI]

    subgraph Data Pipeline
        B --> C[Input Validation<br><i>validator.py</i>]
        C --> D[Preprocessing<br><i>missing values and outliers</i>]
        D --> E[Feature Engineering<br><i>encode, scale, split</i>]
    end

    subgraph Machine Learning
        E --> F[sklearn Pipeline<br><i>ColumnTransformer</i>]
        F --> G[Best Model<br><i>XGBoost/LightGBM/etc.</i>]
        G --> H[Prediction]
    end

    subgraph Explainability And Logging
        H --> I[Prediction History<br><i>predictions.csv</i>]
        H --> J[SHAP Explanation<br><i>explain.py</i>]
        J --> K[Result Display]
    end

    subgraph Experiment Tracking
        L[MLflow] -.->|Tracks| F
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

1. A user enters an academic and demographic profile through Streamlit or the
   FastAPI REST endpoint.
2. Inputs are validated against rules in `src/validator.py`.
3. A saved scikit-learn pipeline scales numerical features and encodes
   categorical, ordinal, and binary features.
4. The trained regression model predicts admission probability.
5. Optional SHAP values explain the most influential transformed features.
6. Input and prediction metadata are appended to `reports/predictions.csv`.

This architecture is designed for a synthetic-data demonstration project. It is
not a real admissions decision system.
