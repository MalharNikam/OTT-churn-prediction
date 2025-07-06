import shap
import pickle

# Load the model and preprocessor
model = pickle.load(open('churn_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Extract trained Random Forest from pipeline
rf_model = model.named_steps['classifier']

# Prepare SHAP Explainer
explainer = shap.TreeExplainer(rf_model)

# Save SHAP Explainer
with open('shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

print("✅ SHAP explainer saved successfully!")
