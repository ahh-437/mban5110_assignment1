from django.conf import settings
from django.shortcuts import render
import pandas as pd

def results_page(request):
    folder = settings.BASE_DIR / "outputs"
    q1 = pd.read_csv(folder / "q1_summary.csv")
    q2 = pd.read_csv(folder / "q2_probabilities.csv")
    q3 = pd.read_csv(folder / "q3_all_test_decisions.csv")
    scores = pd.read_csv(folder / "q3_forecast_scores.csv")
    choices = pd.read_csv(folder / "q3_choice_score.csv")
    updates = pd.read_csv(folder / "q3_information_update.csv")
    sensitivity = pd.read_csv(folder / "q3_no_purchase_sensitivity.csv")
    context = {
        "q1_table": q1.round(4).to_html(index=False),
        "q2_table": q2.round(4).to_html(index=False),
        "q3_table": q3.round(2).to_html(index=False),
        "forecast_table": scores.round(4).to_html(index=False),
        "choice_table": choices.round(4).to_html(index=False),
        "update_table": updates.round(2).to_html(index=False),
        "sensitivity_table": sensitivity.round(2).to_html(index=False)
    }
    return render(request, "analysis/results.html", context)