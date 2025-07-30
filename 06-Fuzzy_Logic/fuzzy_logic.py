#!/usr/bin/env python3
"""
A simple fuzzy inference system (FIS) example to control a fan's speed
based on temperature and humidity. This implementation covers the main
steps:

1. Fuzzification: Calculate degrees of membership for input values.
2. Rule Evaluation: Use fuzzy IF-THEN rules (using min for AND).
3. Aggregation: Combine fuzzy outputs using the max operator.
4. Defuzzification: Convert the aggregated fuzzy output into a crisp value
   using the centroid method.

Author: Aashiq Edavalapati
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Fuzzification Functions for Inputs
# ------------------------------

def cold_temperature(temp):
    """Fuzzy membership function for 'Cold' temperature (°C)."""
    if temp <= 15:
        return 1.0
    elif 15 < temp < 25:
        return (25 - temp) / 10.0
    else:
        return 0.0

def moderate_temperature(temp):
    """Fuzzy membership function for 'Moderate' temperature (°C)."""
    if 15 < temp < 25:
        return (temp - 15) / 10.0
    elif 25 <= temp < 35:
        return (35 - temp) / 10.0
    else:
        return 0.0

def hot_temperature(temp):
    """Fuzzy membership function for 'Hot' temperature (°C)."""
    if temp <= 25:
        return 0.0
    elif 25 < temp < 35:
        return (temp - 25) / 10.0
    else:
        return 1.0

def low_humidity(hum):
    """Fuzzy membership function for 'Low' humidity (%)."""
    if hum <= 30:
        return 1.0
    elif 30 < hum < 50:
        return (50 - hum) / 20.0
    else:
        return 0.0

def high_humidity(hum):
    """Fuzzy membership function for 'High' humidity (%)."""
    if hum <= 30:
        return 0.0
    elif 30 < hum < 50:
        return (hum - 30) / 20.0
    else:
        return 1.0

# ------------------------------
# Membership Functions for Output (Fan Speed)
# Scale: 0 (slow) to 100 (fast)
# ------------------------------

def low_speed(x):
    """Fuzzy membership function for 'Low' fan speed."""
    if x < 0 or x > 50:
        return 0.0
    elif x <= 25:
        return x / 25.0
    else:
        return (50 - x) / 25.0

def medium_speed(x):
    """Fuzzy membership function for 'Medium' fan speed."""
    if x < 25 or x > 75:
        return 0.0
    elif x <= 50:
        return (x - 25) / 25.0
    else:
        return (75 - x) / 25.0

def high_speed(x):
    """Fuzzy membership function for 'High' fan speed."""
    if x < 50 or x > 100:
        return 0.0
    elif x <= 75:
        return (x - 50) / 25.0
    else:
        return (100 - x) / 25.0

# ------------------------------
# Fuzzy Inference and Defuzzification
# ------------------------------

def fuzzy_inference(temp, hum):
    """
    Evaluate the fuzzy inference system for given temperature and humidity.
    The rule base is defined as follows:
    
      Rule 1: IF temperature IS cold THEN fan_speed IS low
      Rule 2: IF temperature IS moderate AND humidity IS high THEN fan_speed IS low
      Rule 3: IF temperature IS moderate AND humidity IS low THEN fan_speed IS medium
      Rule 4: IF temperature IS hot THEN fan_speed IS high
    """
    # Step 1. Fuzzification for temperature
    mu_cold = cold_temperature(temp)
    mu_moderate = moderate_temperature(temp)
    mu_hot = hot_temperature(temp)
    
    # Fuzzification for humidity
    mu_low_hum = low_humidity(hum)
    mu_high_hum = high_humidity(hum)
    
    # Debug print for membership degrees
    print(f"Temperature={temp}°C -> Cold: {mu_cold}, Moderate: {mu_moderate}, Hot: {mu_hot}")
    print(f"Humidity={hum}% -> Low: {mu_low_hum}, High: {mu_high_hum}")
    
    # Step 2. Rule Evaluation using the minimum operator for AND
    
    # Rule 1: IF temperature IS cold THEN fan_speed IS low
    rule1 = mu_cold
    
    # Rule 2: IF temperature IS moderate AND humidity IS high THEN fan_speed IS low
    rule2 = min(mu_moderate, mu_high_hum)
    
    # Rule 3: IF temperature IS moderate AND humidity IS low THEN fan_speed IS medium
    rule3 = min(mu_moderate, mu_low_hum)
    
    # Rule 4: IF temperature IS hot THEN fan_speed IS high
    rule4 = mu_hot
    
    print(f"Rule activations -> R1: {rule1}, R2: {rule2}, R3: {rule3}, R4: {rule4}")
    
    # Step 3. Aggregation of outputs:
    # We use a clipping approach: for each rule, clip the output membership functions
    # and then take the maximum (union) over rules.
    
    # Define the output domain for fan speed
    x = np.linspace(0, 100, 500)
    aggregated = np.zeros_like(x)
    
    # For each rule, compute the clipped membership function over fan speed domain.
    low_rule1 = np.array([min(rule1, low_speed(val)) for val in x])
    low_rule2 = np.array([min(rule2, low_speed(val)) for val in x])
    med_rule3 = np.array([min(rule3, medium_speed(val)) for val in x])
    high_rule4 = np.array([min(rule4, high_speed(val)) for val in x])
    
    # Aggregate: pointwise maximum of all rule contributions
    for contribution in [low_rule1, low_rule2, med_rule3, high_rule4]:
        aggregated = np.maximum(aggregated, contribution)
    
    # Step 4. Defuzzification using the centroid (center-of-gravity) method
    if np.sum(aggregated) == 0:
        # Avoid division by zero: if no rule fired, return a default value.
        crisp_value = 0
    else:
        crisp_value = np.sum(x * aggregated) / np.sum(aggregated)
    
    # Optionally, plot the aggregated output membership function.
    plt.figure(figsize=(8, 4))
    plt.plot(x, aggregated, label='Aggregated Output')
    plt.title('Aggregated Fuzzy Output for Fan Speed')
    plt.xlabel('Fan Speed')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return crisp_value

# ------------------------------
# Main function for testing
# ------------------------------

if __name__ == '__main__':
    # Example input values:
    temperature_input = 29   # in °C
    humidity_input = 40      # in %
    
    fan_speed = fuzzy_inference(temperature_input, humidity_input)
    print(f"Defuzzified Fan Speed: {fan_speed:.2f} (out of 100)")
