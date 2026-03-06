# Mock Banking Database
# In a real app, this would be a SQL database
USERS_DB = {
    "user_123": {
        "name": "Alex Johnson",
        "account_type": "Savings",
        "balance": 15420.50,
        "currency": "USD",
        "credit_score": 750,
        "active_loans": []
    }
}

def get_balance(user_id="user_123"):
    """
    Retrieves the current account balance for a user.
    """
    user = USERS_DB.get(user_id)
    if not user:
        return "User not found."
    return f"${user['balance']:,} {user['currency']}"

def check_loan_eligibility(amount: int, user_id="user_123"):
    """
    Checks if a user is eligible for a loan based on credit score and balance.
    """
    user = USERS_DB.get(user_id)
    if not user:
        return "User not found."
    
    if user['credit_score'] < 600:
        return "Sorry, your credit score is too low for a loan at this time."
    
    max_loan = user['balance'] * 5  # Simple rule: max loan is 5x balance
    
    if amount > max_loan:
        return f"You are eligible for a maximum loan of ${max_loan:,}. The requested amount is too high."
    
    return f"Congratulations! You are eligible for a loan of ${amount:,} with an interest rate of 5.5% APR."
