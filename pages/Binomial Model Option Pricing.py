import numpy as np
import streamlit as st
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Formatting Page
st.title('Binomial Model Option Pricing')

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

#sns.lineplot(x=finalprices, y=finalprobas, color='r')
#plt.title("Probability Distribution of Expiration Price")
#plt.xlabel("Underlying Price ($)")
#plt.ylabel("Probability")

#st.pyplot()

# Explaining the Binomial Model

st.header("Comments on the  Binomial Model")

st.markdown("""
The Binomial model for option pricing is a non-continuous time model that uses principles from the binomial distribution.
The model makes a number of assumptions including:
- The underlying price can only move in two particular ways, upwards by factor u or downwards by factor d.
- The probability of upwards and downwards movement are constant, and price returns are log-normal.
- The risk-free rate (r) is constant.
- No arbitrage, such that the expected underlying price through time should equal that of the risk-free return
- Time is discretised.
- Options are European.

The no-arbitrage condition implies that there are no risk-free strategies that earn more that the risk-free rate that is pre-determined.
In an environment where arbitrage exists, traders could open positions that are hedged against price movement risk (delta-neutral) and expect positive returns over time.
A simple way to profit off this would be to buy in the current market and sell in the futures market  (if the futures market price is inflated), we would then make a risk-free profit by expiration as the prices meet eachother.
The no-arbitrage condition ensures that options are fairly priced by isolating the role of volatility.

To account for this no-arbitrage condition, we must operate in a risk-neutral probability space.
This means that we must determine up and down factors, as well as up and down probabilities such that the discounted expected underlying price through time is equal to its current price.
To do this, we assume a log-normal distribution of returns through time, which leads us to the following equations:
""")
st.latex("u  = e^{σΔt^{0.5}}, d = e^{-σΔt^{0.5}}")
st.latex(r"P(u) = \frac{e^{-rT} - d}{u-d}")
st.latex("P(d) = 1 - P(u)")

st.write("""
Where σ = annualized volatility, Δt = timestep, r = risk-free rate and T = time (annualised).

This model finds strength in its ease of application and understanding, providing an efficient way of getting an option price.
It is not computationally expensive and yields an approximation quickly.

However, it suffers from numerous weaknesses too.
First, the assumptions around the constant volatility and risk-free rates are dangerous, as they can change through time and significantly affect option prices.
Next, the simple fact that volatility is assumed makes models in general weak.
The use of discretised time can also lead to computational inaccuracies.
The log-normal assumption is unlikely to be true, especially in market events causing fatter tails than expected by such a return distribution.
The model loses in  accuracy in high volatility and short-time environments.
Finally, they do not account for transaction costs and liquidity, which may severely affect traders.""")



