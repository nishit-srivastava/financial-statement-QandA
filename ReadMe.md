# setting up environment
# conv_ai_financial_qa# From inside conv_ai_financial_qa/
python -m venv .venv

# Activate venv (Windows PowerShell)
.venv\Scripts\Activate


pip install ipykernel
python -m ipykernel install --user --name=conv_ai_env

pip install -r requirements.txt

streamlit run app.py