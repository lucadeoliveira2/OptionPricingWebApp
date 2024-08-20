import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Creating the pages

st.title("Black-Scholes Option Pricing")
#st.set_option('deprecation.showPyplotGlobalUse', False)

# Implementing BS formula

def blackScholes(S, K, T, vol, r):

    d1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*T**0.5)
    d2 = d1 - vol*(T**0.5)
    priceC = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    priceP = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)

    return(priceC, priceP)

# Displaying option prices

st.subheader("Option Prices")
st.write("The price of the options are calculated using the following Black-Scholes equations:")
st.latex("C = N(d1)*S - N(d2)*K*exp(-rT)")
st.latex("P = N(-d2)*K*exp(-rT) - N(-d1)*S""")
st.write("where")
st.latex(r'''d1 = \left(\frac{log(S/K) +(r + \left(\frac{o^{2}}{2}\right))T}{oT^{0.5}}\right)''')
st.latex("d2 = d1 - oT^{0.5}")
st.write("with S = Current Price, K = Exercise Price, T = Time to Expiry (in Years), r = Interest Rate, o = Volatility")

# Getting input data from user

st.subheader("Inputs")

col1, col2, col3 = st.columns(3)
S = col1.number_input("Current Price", min_value=0.0, value=100.0, step=0.01)
K = col2.number_input("Exercise Price", min_value=0.0, value=102.0, step = 0.01)
T = col3.number_input("Days to Expiry", min_value=0, value=30)/365 # Has to be divided by 365
vol = col1.number_input("Volatility (%)", min_value=0.0, value=12.0, step=0.01)/100
r = col3.number_input("Interest Rate (%)", min_value=0.0, value=4.5, step=0.01)/100

priceC, priceP = blackScholes(S, K, T, vol, r)

# Displaying Prices

st.subheader("Theoretical Prices")
st.metric(label="Call Price", value=f"${priceC:,.3f}")
st.metric(label="Put Price", value=f"${priceP:,.3f}")

# Implied Vol Calculator
st.subheader("Implied Volatility Calculator")
col1, col2, col3 = st.columns(3)
paidprice = col1.number_input(label='Option Cost ($)', min_value=0.000, value=1.000, step=0.001)
optionvega = col2.number_input(label='Option Vega (Absolute Value)', min_value=0.000, value=0.500, step=0.001)
optiontype2 = col3.selectbox(label="Option Type ", options=['Call ', 'Put '])
if optiontype2 == "Call ":
    theoreticalprice = priceC
else:
    theoreticalprice = priceP
pricediff = paidprice - theoreticalprice
volatilitypremium = pricediff/optionvega
impliedvolatility = vol*100 + volatilitypremium

st.metric(label=f"IV of the {optiontype2} option:", value=f"{impliedvolatility:.2f}%")

# Heatmap Inputs

st.subheader("Price Heatmaps")
st.write("(Figures may struggle to render, try multiple times if needed)")
col1, col2 = st.columns(2)
volrange = col1.number_input(label="Volatility Range (%)", min_value=0, max_value=100, value=50)
Srange = col2.number_input(label="Underlying Price Range (%)", min_value=0, max_value=100, value=5)/100

# Heatmap

lowvol = vol*100 - vol*volrange # in %
highvol = vol*100 + vol*volrange
volatilities = np.linspace(lowvol, highvol, 10)
lowprice = S - S*Srange
highprice = S + S*Srange
underlyingprices = np.linspace(lowprice, highprice, 10)

pricedf = [0 for _ in range(100)]
pricedf = pandas.DataFrame(np.array(pricedf).reshape((10, 10)))

for i in range(len(underlyingprices)):
    Scurrent = underlyingprices[i]
    for j in range(len(volatilities)):
        volcurrent = volatilities[j]/100
        pricedf.iloc[j, i] = blackScholes(Scurrent, K, T, volcurrent, r)[0]

axes = plt.subplots(ncols=2, nrows=1, figsize = (15,7))[1]
cmap = LinearSegmentedColormap.from_list('Heat', ['crimson', 'orange', 'yellow', 'lime']) # Creating colormap
sns.heatmap(pricedf, annot=True, annot_kws={"size":10}, fmt='.2f', cmap=cmap, ax=axes[0])
axes[0].set_xticks(np.arange(len(underlyingprices)) + 0.5,
           labels=np.round(underlyingprices, 2), rotation=45)
axes[0].set_yticks(np.arange(len(volatilities)) + 0.5,
           labels=np.round(volatilities, 2), rotation=45)
axes[0].set_xlabel("Underlying Price ($)")
axes[0].set_ylabel("Volatility (%) ")
axes[0].set_title("Call Option Price ($)")

for i in range(len(underlyingprices)):
    Scurrent = underlyingprices[i]
    for j in range(len(volatilities)):
        volcurrent = volatilities[j]/100
        pricedf.iloc[j, i] = blackScholes(Scurrent, K, T, volcurrent, r)[1]

sns.heatmap(pricedf, annot=True, annot_kws={"size":10}, fmt='.2f', cmap=cmap, ax=axes[1])
axes[1].set_xticks(np.arange(len(underlyingprices)) + 0.5,
           labels=np.round(underlyingprices, 2), rotation=45)
axes[1].set_yticks(np.arange(len(volatilities)) + 0.5,
           labels=np.round(volatilities, 2), rotation=45)
axes[1].set_xlabel("Underlying Price ($)")
axes[1].set_ylabel("Volatility (%) ")
axes[1].set_title("Put Option Price ($)")
# st.set_option('deprecation.showPyplotGlobalUse', False) # Removing warning
st.pyplot()

# P&L charts

st.subheader("P&L calculator")
col1, col2 = st.columns(2)
optioncost = col1.number_input(label="Option cost ($)", min_value=0.00, value=1.00, step=0.01)
optiontype= col2.selectbox(label="Option Type", options=['Call', 'Put'])
volrange = col1.number_input(label="Realised Volatility Range (%)", min_value=0, max_value=100, value=50)
pricerange = col2.number_input(label="Price Range (%)", min_value=0, max_value=100, value=50)

minvol = vol*100 - vol*volrange
maxvol = vol*100 + vol*volrange
minprice = optioncost - optioncost*(pricerange/100)
maxprice = optioncost + optioncost*(pricerange/100)
volatilities = np.linspace(minvol, maxvol, 10)
prices = np.linspace(minprice, maxprice, 10)

theoreticaldf = [0 for _ in range(100)]
theoreticaldf = pandas.DataFrame(np.array(theoreticaldf).reshape((10,10)))

for i in range(len(prices)):
    paidprice = prices[i]
    for j in range(len(volatilities)):
        realisedvol = volatilities[j]/100
        if optiontype == "Call":
            theoreticalprice = blackScholes(S, K, T, realisedvol, r)[0]
        else:
            theoreticalprice = blackScholes(S, K, T, realisedvol, r)[1]
        theoreticaledge = theoreticalprice - paidprice
        theoreticaldf.iloc[j, i] = theoreticaledge

plt.figure(figsize=(12,8))
cmap = LinearSegmentedColormap.from_list('Heat', ['crimson', 'orange', 'yellow', 'lime'])
sns.heatmap(theoreticaldf, annot=True, annot_kws={"size":10}, fmt=".2f", cmap=cmap, center=0)
plt.xticks(ticks=np.arange(len(prices))+0.5, labels=np.round(prices,2), rotation=45)
plt.yticks(ticks=np.arange(len(volatilities))+0.5, labels=np.round(volatilities,2), rotation=45)
plt.xlabel("Paid Price ($)")
plt.ylabel("Realised Volatility (%)")
plt.title(f"Black-Scholes Theoretical Edge for the {optiontype} Option")
st.pyplot()