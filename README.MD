# MBAN 5110 Assignment 1

This project contains my analysis of the bike sales data and a Django page showing the results. I used OLS to forecast demand and logistic regression to estimate product choices. I then used these models to choose prices and quantities within the reserved capacity.

## Files

- `Assignment1.ipynb`: notebook with the code, results and explanations.
- `Assignment1.py`: Python version of the analysis.
- `Assignment1.html`: HTML version of the notebook.
- `sales_transactions.csv`: dataset used in the assignment.
- `outputs/`: result tables and charts.
- `manage.py`: starts the Django project.
- `requirements.txt`: packages needed to run the project.

## Running the project

Open Anaconda Prompt, activate the environment, and go to the folder containing `manage.py`. Run:

```bash
python -m pip install -r requirements.txt
python Assignment1.py
python manage.py check
python manage.py migrate
python manage.py runserver
```

Close each chart window if the analysis pauses. Once the server starts, open http://127.0.0.1:8000/ in a browser. Press Ctrl+C in the terminal to stop it.

The notebook can also be run from the same folder as `sales_transactions.csv`. The Django page displays the saved results in `outputs/`; it does not run the analysis each time the page opens.

## Forecasts and capacity

I grouped deliveries into two-week periods. Forecast horizons are measured before the start of each selling period, using only sales periods completed by the decision date.

For Question 3, I used four two-week periods to represent two months. Capacity is reserved 56 days before selling, and prices and quantities are chosen 14 days before selling. Following the professor's clarification, capacity equals the horizon-4 forecast, set to zero if negative and rounded to a whole bike, with half units rounded upward. Prices and quantities are optimized within this fixed capacity.

The test set contains 26 periods starting January 27, 2025. The Question 3 models are trained using information available by December 2, 2024. Their coefficients stay fixed, while newer completed sales become available as forecast inputs.

## Pricing assumptions

I compared 125 price combinations based on the training data and selected the plan with the highest modeled revenue. Costs were not provided, so the calculation does not measure profit. Existing orders keep their original quantities and prices. There are no recorded advance bookings at the two-week decision dates in the test set.

The data only includes purchases. I assumed a 50% purchase rate at the reference median prices and a price-response value of 2. I also tested purchase rates of 25%, 50% and 75%, with response values of 1, 2 and 3. These are assumptions, not estimates from the data. The forecast is calibrated to reproduce purchases at the reference prices, with average order sizes assumed stable.

The recommended prices and quantities depend on these assumptions. The analysis does not establish the causal effect of changing prices or show the revenue that would actually have been earned. The notebook explains the method, sensitivity results and limitations in more detail.
