import numpy as np
import streamlit as st
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Formatting Page
st.title('Option Pricing Web App')
st.header('Binomial Model Option Pricing')

# Getting Option Info From End User

col1, col2, col3 = st.columns(3)
S = col1.number_input("Current Price", min_value=0.0, value=100.0, step=0.01)
K = col2.number_input("Exercise Price", min_value=0.0, value=102.0, step=0.01)
days = col3.number_input("Days to Expiry", min_value=0, value=30)
vol = col1.number_input("Volatility (%)", min_value=0.0, value=12.0, step=0.01)/100
r = col2.number_input("Interest Rate (%)", min_value=0.0, value=4.5, step=0.01)/100
steps = col3.number_input("Number of Steps", min_value=0, max_value=1000, value=100, step=1)
optiontype = col2.selectbox("Option Type", options=('Call', 'Put'))

# Calculating Other Variables

T = days/365
timestep = T/steps # timestep between each price movement
u = np.exp(vol*np.sqrt(timestep)) # Up Factor
d = 1/u # Down Factor
Pu = (np.exp(r*timestep) - d)/(u-d) # Up Probability
Pd = 1 - Pu # Down Probability

# Making Price Return and Path Probability Function

def PathArrays(steps, S):

    ends = steps + 1 # Although there are 2**steps possible paths, there are only ends possible final prices
    finalprices = np.zeros(ends, dtype=float)
    finalprobas = np.zeros(ends, dtype=float)

    for k in range(ends):

        k = k # Number of downs in path
        n = steps - k # Number of ups in path
        choose = math.factorial(steps)/(math.factorial(n)*math.factorial(steps-n)) # Accounting for fact that many paths lead to same result

        finalprices[k] = (u**n) * (d**k) * S
        finalprobas[k] = (Pu**n) * (Pd**k) * choose

    expectedprice = np.sum(finalprices*finalprobas)

    return finalprices, finalprobas, expectedprice

finalprices, finalprobas, expectedprice = PathArrays(steps, S)
st.subheader(f"The expected expiration price of the underlying is {expectedprice: .2f}$")

# Computing Option Value

def OptionValue(finalprices, finalprobas, r, K, T, optiontype):

    optiondict = {
        'Call': 1,
        'Put': -1
    }
    optionfactor = optiondict[optiontype] # Accounting for type of option

    pathvalue = (finalprices - K) * optionfactor # expected end value for each path
    pathpayoff = np.where(pathvalue > 0, pathvalue, 0) # payoff for each path
    expectedpayoff = np.sum(pathpayoff * finalprobas) # expected payoff considering probabilities
    discountedpayoff = expectedpayoff * np.exp(-r * T) # discounting to today's value

    return discountedpayoff

optionvalue = OptionValue(finalprices, finalprobas, r, K, T, optiontype)
st.subheader(f"Theoretical value of the {optiontype} option is {optionvalue:.3f}$")

# Plotting Expiration Price

sns.lineplot(x=finalprices, y=finalprobas, color='r')
plt.title("Probability Distribution of Expiration Price")
plt.xlabel("Underlying Price ($)")
plt.ylabel("Probability")

st.pyplot()



