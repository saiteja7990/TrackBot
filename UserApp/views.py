from django.shortcuts import render
from Database import getConnection
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import re
from django.core.mail import EmailMultiAlternatives
from personal_finance_manager.settings import DEFAULT_FROM_EMAIL
from django.conf import settings
from sklearn.linear_model import LinearRegression
import numpy as np


# Create your views here.
def index(request):
    return render(request,'index.html')


def RegAction(request):
    n = request.POST['name']
    e = request.POST['email']
    m = request.POST['mobile']
    a = request.POST['Address']
    u = request.POST['username']
    p = request.POST['password']

    con = getConnection()
    cur = con.cursor()
    cur.execute("select * from user where email='"+e+"'")
    d = cur.fetchone()
    if d is None:
        i=cur.execute("insert into user values(null,'"+n+"','"+e+"','"+m+"','"+a+"','"+u+"','"+p+"')")
        print(i)
        con.commit()
        context={'msg':'Registration Successful..!!'}
        return render(request, 'index.html',context)
    else:
        context = {'msg': 'Email Already Exist..!!'}
        return render(request, 'index.html', context)


def LogAction(request):
    u = request.POST['uname']
    p = request.POST['pwd']

    con = getConnection()
    cur = con.cursor()
    cur.execute("select * from user where username='" + u + "' and password='"+p+"'")
    d = cur.fetchone()
    if d is None:
        context={'lmsg':'Login Failed..!!'}
        return render(request, 'index.html',context)
    else:
        email= d[2]
        request.session['email']=email
        request.session['username'] = u
        return render(request, 'UserHome.html')

def home(request):
    return render(request, 'UserHome.html')

def AddExpenses(request):
    df = pd.read_csv('Dataset/transactions.csv')

    s1=""
    for i in df['category'].unique():
        s1+=f"<option>{i}</option>"
    s1 += ""
    s2 = ""
    for i in df['type'].unique():
        s2 += f"<option>{i}</option>"
    s2 += ""

    s3 = ""
    for i in df['payment_method'].unique():
        s3 += f"<option>{i}</option>"
    s3 += ""

    s4 = ""
    for i in df['budget_category'].unique():
        s4 += f"<option>{i}</option>"
    s4 += ""

    context={'s1':s1,'s2':s2,'s3':s3,'s4':s4}
    return render(request,'AddExpenses.html',context)

def AddEIAction(request):
    user_id = request.POST['user_id']
    date_str = request.POST['date']  # comes as YYYY-MM-DD from form
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    formatted_date = date_obj.strftime('%d-%m-%Y')

    category = request.POST['category']
    desc = request.POST['desc']
    amount = float(request.POST['amount'])  # make sure it's numeric
    type = request.POST['type']
    balance = request.POST['balance']
    merchant = request.POST['merchant']
    p_m = request.POST['p_m']
    b_c = request.POST['b_c']
    note = request.POST['note']

    file_path = os.path.join(settings.BASE_DIR, "Dataset", "transactions.csv")
    existing_data = pd.read_csv(file_path)

    # --- Fix: ensure date column is always string dd-mm-yyyy ---
    existing_data["date"] = pd.to_datetime(existing_data["date"], dayfirst=True, errors="coerce").dt.strftime(
        '%d-%m-%Y')

    # --- Extract month-year from entered date ---
    target_month = date_obj.strftime('%B %Y')  # Example: "August 2025"
    existing_data["month"] = pd.to_datetime(existing_data["date"], dayfirst=True, errors="coerce").dt.strftime('%B %Y')

    # --- Calculate current total for that month ---
    month_total = existing_data.loc[existing_data["month"] == target_month, "amount"].sum()

    # --- Check limit ---
    if month_total + amount > 200000:

        email = request.session['email']

        subject = "Limit Cross Warning"
        text_content = ""
        html_content = (
            f"Monthly limit exceeded! Current total: ₹{month_total:.2f}, adding {amount} would exceed ₹200000.")

        from_mail = DEFAULT_FROM_EMAIL
        to_mail = [email]
        msg = EmailMultiAlternatives(subject, text_content, from_mail, to_mail)
        msg.attach_alternative(html_content, "text/html")
        if msg.send():
            sts = 'sent'
            print(sts)
        return render(request, 'AddExpenses.html', {
            'msg': f"Monthly limit exceeded! Current total: ₹{month_total:.2f}, adding {amount} would exceed ₹200000."})

    else:
        # Add new row
        new_entry = pd.DataFrame([{
            'user_id': user_id,
            'date': formatted_date,  # always dd-mm-yyyy
            'category': category,
            'description': desc,
            'amount': amount,
            'type': type,
            'balance': balance,
            'merchant': merchant,
            'payment_method': p_m,
            'budget_category': b_c,
            'notes': note,
            'month': target_month  # keep month column consistent
        }])

        # Insert as first row
        updated_data = pd.concat([new_entry, existing_data], ignore_index=True)

        # --- Fix: force all dates to dd-mm-yyyy before saving ---
        updated_data["date"] = pd.to_datetime(updated_data["date"], dayfirst=True, errors="coerce").dt.strftime(
            '%d-%m-%Y')

        updated_data.to_csv(file_path, index=False, mode='w')

        print(f"Added successfully! New total for {target_month} is ₹{month_total + amount:.2f}.")
        return render(request, 'AddExpenses.html',{'msg': f"Added successfully! New total for {target_month} is ₹{month_total + amount:.2f}."})


def Upload(request):
    return render(request, 'UploadDataset.html')

global df
def UploadAction(request):
    global df
    if request.method == 'POST' and request.FILES.get('dataset'):
        csv_file = request.FILES['dataset']
        # Read file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # (Optional) Do something with the data
        print(df.head())  # Just print the first 5 rows in terminal for now

    return render(request,'UploadDataset.html',{'msg':'Dataset Uploaded Successfully..!!'})

def ViewDataset(request):
    global df

    df = pd.read_csv("Dataset/transactions.csv")
    tabledata = "<table id='example' class='table table-striped table-bordered'>"

    # Header
    tabledata += "<thead><tr>"
    for col in df.columns:
        tabledata += f"<th>{col}</th>"
    tabledata += "</tr></thead>"

    # Body
    tabledata += "<tbody>"
    for _, row in df.head(1000).iterrows():
        tabledata += "<tr>"
        for item in row:
            tabledata += f"<td>{item}</td>"
        tabledata += "</tr>"
    tabledata += "</tbody>"

    tabledata += "</table>"

    return render(request, 'Summery.html', {'tabledata': tabledata})



def monthly(request):
    dataf = pd.read_csv("Dataset/transactions.csv")

    # Parse date column
    dataf['date'] = pd.to_datetime(dataf['date'], errors='coerce')
    dataf.dropna(subset=['date'], inplace=True)

    # Filter only 'expense' type rows
    spending = dataf[dataf['type'] == 'expense'].copy()

    # Add a month column for grouping
    spending['month'] = spending['date'].dt.to_period('M').astype(str)

    # Group by month and category, sum the amount
    monthly_spending = spending.groupby(['month', 'category'])['amount'].sum().unstack().fillna(0)

    # Plotting
    plt.figure(figsize=(10, 6))
    for category in monthly_spending.columns:
        plt.plot(monthly_spending.index.values, monthly_spending[category].values.flatten(), marker='o', label=category)

    plt.title('Monthly Spending Trends by Category')
    plt.xlabel('Month')
    plt.ylabel('Amount Spent')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig('Static/monthly_spending_trends.png')
    plt.close()
    return render(request, 'MonthlySpending.html')

def monthly_income(request):
    df = pd.read_csv("Dataset/transactions.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')
    monthly_data = df.groupby(['month', 'type'])['amount'].sum().unstack().fillna(0)

    plt.figure(figsize=(15, 6))
    monthly_data.plot(kind='bar', color=['red', 'green'])
    plt.title('Monthly Income vs Expenses')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('Static/monthly_income_vs_expenses.png')
    plt.close()
    return render(request, 'MonthlyIncome_Expenses.html')

def Cumulative(request):
    df = pd.read_csv("Dataset/transactions.csv")

    # Convert 'date' to datetime safely
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df_sorted = df.sort_values(by='date')

    plt.figure(figsize=(10, 6))

    dates = df_sorted['date'].values.flatten()
    balances = df_sorted['balance'].values.flatten()

    plt.plot(dates, balances, marker='o', linestyle='-')
    plt.title('Cumulative Account Balance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.xticks(rotation=45)
    # Save plot
    os.makedirs('Static', exist_ok=True)
    plt.savefig('Static/cumulative_account_balance.png')
    plt.close()
    return render(request, 'Cumulative.html')

def expense_chart(request):
    df = pd.read_csv("Dataset/transactions.csv")

    # Group by category and sum amounts
    category_sum = df.groupby("category")["amount"].sum()

    # Calculate percentage
    category_percentage = (category_sum / category_sum.sum()) * 100

    # Convert to DataFrame
    result = category_percentage.reset_index()
    result.columns = ["Category", "Percentage"]

    # Convert to list of dicts for template
    data = result.to_dict(orient="records")

    return render(request, "expense_chart.html", {"data": data})


def Chatbot(request):
    return render(request,'Chatbot.html')

df = pd.read_csv("Dataset/transactions.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')

def chatbot_response(user_input):
    question = user_input.lower()


    if "hi" == question:
        response="Hi User! Welcome to Finance Chatbot..!!"
        return response

    if any(phrase in question for phrase in ["how are you", "how r u", "how's it going", "what's up"]):
        response="I'm just a bot, but I'm running smoothly! How can I help you with your finances today?"
        return response

    # --- Intent 1: Total spent per category ---
    if "total" in question and "category" in question:
        totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
        response = "Total spent per category:<br>"
        for cat, amt in totals.items():
            response += f"{cat}: ₹{amt:.2f}<br>"
        return response

    # --- Intent 2: Total spent per month ---
    if "total" in question and "month" in question:
        df["month"] = df["date"].dt.strftime('%B %Y')
        monthly = df.groupby("month")["amount"].sum().sort_values(ascending=False)
        response = "Total spent per month:<br>"
        for month, amt in monthly.items():
            response += f"{month}: ₹{amt:.2f}<br>"
        return response

    # --- Intent 3A: Day-wise spending summary ---
    if any(x in question for x in ["day wise", "weekday", "per day"]):
        df["day"] = df["date"].dt.day_name()
        daywise = df.groupby("day")["amount"].sum().sort_values(ascending=False)
        response = "Total spent per day of week:<br>" + "<br>".join(
            f"{day}: ₹{amt:.2f}" for day, amt in daywise.items()
        )
        return response

    # --- Intent 3B: Spending on a specific weekday ---
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for wd in weekdays:
        if wd in question:
            df["day"] = df["date"].dt.day_name().str.lower()
            matches = df[df["day"] == wd]
            if matches.empty:
                return f"No expenses found on {wd.capitalize()}."

            total = matches["amount"].sum()
            details = "<br>".join(
                f"{row['date'].date()} - ₹{row['amount']} ({row['merchant']})"
                for _, row in matches.iterrows()
            )
            return f"Total spent on {wd.capitalize()}: ₹{total:.2f}<br>{details}"

    # --- Intent 4: Total spent on a specific date ---
    if "date" in question or re.search(r"\d{2}-\d{2}-\d{4}", question):
        date_match = re.search(r"\d{2}-\d{2}-\d{4}", question)  # DD-MM-YYYY
        if date_match:
            specific_date = pd.to_datetime(date_match.group(), dayfirst=True, errors='coerce')
            matches = df[df["date"].dt.date == specific_date.date()]
            if not matches.empty:
                total = matches["amount"].sum()
                response = f"Total spent on {specific_date.strftime('%d-%m-%Y')}: ₹{total:.2f}<br>"
                for _, row in matches.iterrows():
                    response += f" - {row['merchant']} ({row['category']}): ₹{row['amount']}<br>"
                return response
            else:
                return f"No expenses found on {specific_date.strftime('%d-%m-%Y')}."
        return "Please provide a valid date in DD-MM-YYYY format."

    # --- Intent 5: Filter by merchant ---
    if "merchant" in question:
        merchants = df["merchant"].dropna().str.lower().unique()
        for merchant in merchants:
            if merchant in question:
                result = df[df["merchant"].str.lower() == merchant].sort_values(by="date", ascending=False)
                if not result.empty:
                    response = f"Transactions with merchant '{merchant}':<br>"
                    for _, row in result.iterrows():
                        response += f"{row['date']} - ₹{row['amount']} ({row['category']})<br>"
                    return response
                else:
                    return f"No transactions found for merchant '{merchant}'"

    # --- Intent 6: Filter by amount ---
    if "above" in question or "greater than" in question:
        for word in question.split():
            if word.replace('.', '').isdigit():
                threshold = float(word)
                result = df[df["amount"] > threshold].sort_values(by="amount", ascending=False)
                if not result.empty:
                    response = f"Transactions above ₹{threshold}:<br>"
                    for _, row in result.iterrows():
                        response += f"{row['date']} - ₹{row['amount']} ({row['merchant']})<br>"
                    return response
                else:
                    return f"No transactions above ₹{threshold}"

    # --- Intent 7: Last 5 transactions ---
    if "last" in question and "transactions" in question:
        last = df.sort_values(by="date", ascending=False).head(5)
        response = "Last 5 transactions:<br>"
        for _, row in last.iterrows():
            response += f"{row['date']} - ₹{row['amount']} ({row['merchant']})<br>"
        return response

    # Define list of full month names
    months_list = [datetime(2025, m, 1).strftime('%B') for m in range(1, 13)]

    # --- Intent 8: Total spent in a specific month ---
    if "total" in question.lower() and ("month" in question.lower() or any(month.lower() in question.lower() for month in months_list)):
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df["month"] = df["date"].dt.strftime('%B %Y')

        # Extract possible month and year from question using regex
        month_pattern = r"(" + "|".join(months_list) + r")"
        year_pattern = r"\b(20\d{2})\b"

        found_month_match = re.search(month_pattern, question, re.IGNORECASE)
        found_year_match = re.search(year_pattern, question)

        found_month = found_month_match.group(1).capitalize() if found_month_match else None
        found_year = found_year_match.group(1) if found_year_match else None

        if found_month:
            full_month = f"{found_month} {found_year}" if found_year else found_month
            matches = df[df["month"].str.contains(full_month, na=False)]

            if not matches.empty:
                total = matches["amount"].sum()
                return f"Total spent in {full_month}: ₹{total:.2f}"
            else:
                return f"No data found for {full_month}."
        else:
            # Default: show all monthly totals
            monthly = df.groupby("month")["amount"].sum().sort_values(ascending=False)
            response = "Total spent per month:<br>"
            for month, amt in monthly.items():
                response += f"{month}: ₹{amt:.2f}<br>"
            return response

    return "Sorry, I couldn't understand your query."


def ChatAction(request):


    question = request.GET.get('mytext', '')
    print(question)

    if not question:
        return HttpResponse("Invalid request", content_type="text/plain")

    response = chatbot_response(question)
    return HttpResponse(response, content_type="text/plain")

df = pd.read_csv("Dataset/transactions.csv")

# Parse date column (DD-MM-YYYY format)
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

# Extract month & year
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Group by category and month
monthly_expenses = df.groupby(['category', 'year', 'month'])['amount'].sum().reset_index()

def train_category_model(category):
    data = monthly_expenses[monthly_expenses['category'] == category]

    # Create time index (year*12 + month for sequence)
    data['time'] = data['year'] * 12 + data['month']

    X = data[['time']]
    y = data['amount']

    model = LinearRegression()
    model.fit(X, y)

    return model, data


def predict_future_expense(category):
    model, data = train_category_model(category)

    last_time = data['time'].max()
    future_time = np.array([[last_time + 1]])  # next month

    prediction = model.predict(future_time)[0]
    return round(prediction, 2)


def Predict_Expenses(request):


    category = None
    prediction = None
    categories = monthly_expenses['category'].unique()

    if request.method == "POST":
        category = request.POST.get("category")
        prediction = predict_future_expense(category)

    return render(request, "predict_expense.html", {
        "categories": categories,
        "selected_category": category,
        "prediction": prediction
    })