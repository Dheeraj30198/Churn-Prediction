param(
    [string]$DataPath = "Telco_customer_churn.xlsx",
    [string]$TargetCol = "Churn Value",
    [string]$ArtifactDir = "artifacts"
)

$python = ".\.venv\Scripts\python"

Write-Host "Step 1/3: EDA"
& $python -c "from pathlib import Path; from churn_model import load_data, run_eda; df=load_data(Path('$DataPath')); run_eda(df, '$TargetCol', Path('$ArtifactDir/eda')); print('EDA completed')"

Write-Host "Step 2/3: Model Building + Evaluation + Class Imbalance + Feature Importance"
& $python churn_model.py --data-path "$DataPath" --target-col "$TargetCol" --artifact-dir "$ArtifactDir" --skip-eda --skip-tuning --cv-folds 3

Write-Host "Step 3/3: Improve Model (Hyperparameter Tuning)"
& $python churn_model.py --data-path "$DataPath" --target-col "$TargetCol" --artifact-dir "$ArtifactDir" --skip-eda --cv-folds 3 --tune-iters 10

Write-Host "Done. Check $ArtifactDir/eda, $ArtifactDir/models, and $ArtifactDir/reports."
