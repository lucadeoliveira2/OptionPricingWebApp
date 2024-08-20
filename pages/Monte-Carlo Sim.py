## Monte-Carlo Simulation

import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

# Layout

st.title("Monte-Carlo Option Pricing")
st.write("We can price options using Monte-Carlo simulations, which assumes a risk-neutral and arbitrage-free environment. First, we need to simulate Brownian motion using normal deviates (normally distributed entries). The normal deviates were computed using the Box-Muller method and uniform deviates. Finally, the Euler-Maruyama method was used to compute the path of the underlying price, such that prices change every minute of active trading. Therefore, options that have longer expirations will include more steps in the process.")

col1, col2, col3 = st.columns(3)
S = col1.number_input("Current Price", min_value=0.0, value=100.0, step=0.01)
K = col2.number_input("Exercise Price", min_value=0.0, value=102.0, step=0.01)
T = col3.number_input("Days to Expiry", min_value=0, value=30)/365 # Has to be divided by 365
vol = col1.number_input("Volatility (%)", min_value=0.0, value=12.0, step=0.01)/100
r = col3.number_input("Interest Rate (%)", min_value=0.0, value=4.5, step=0.01)/100

# First we set parameters
T = T  # Time to expiration
N = int(np.floor(T * 252 * 7 * 60))  # Number of steps
dt = T / N  # Timestep
M = st.number_input(label="Number of Simulations", min_value=0, max_value=5000, value=50)  # Number of simulations

# Generate random numbers using the Box-Muller Method
np.random.seed(100)
U1 = np.random.uniform(size=(N, M))
U2 = np.random.uniform(size=(N, M))
R = -2 * np.log(U1)
theta = 2 * np.pi * U2
Z1 = np.sqrt(R) * np.cos(theta)

# Compute Brownian motion
W = np.cumsum(np.sqrt(dt) * Z1, axis=0)

# Apply the Euler-Maruyama method
Sdf = np.zeros((N, M))
Sdf[0, :] = S  # Initial underlying price
for i in range(1, N):
    Sdf[i, :] = Sdf[i-1, :] * (1 + r * dt + vol * (W[i, :] - W[i-1, :]))
Sdf = pandas.DataFrame(Sdf)

# We then plot this

for j in range(M):
    sns.lineplot(Sdf.iloc[:, j])
plt.title("Simulated Underlying Price")
plt.xlabel("Timestep")
plt.ylabel("Price ($)")
st.pyplot()

# Finally, we calculate the expected value of the option and discount it back to today's price

expsum = 0
for j in range(M):
    expvalue = Sdf.iloc[N-1, j] - K
    if expvalue > 0:
        expsum += expvalue
meanval = expsum/M
currentval = meanval*np.exp(-r*T)
st.metric(label="Call Price", value=f"${currentval:,.3f}")

expsum = 0
for j in range(M):
    expvalue = K - Sdf.iloc[N-1, j]
    if expvalue > 0:
        expsum += expvalue
meanval = expsum/M
currentval = meanval*np.exp(-r*T)
st.metric(label="Put Price", value=f"${currentval:,.3f}")