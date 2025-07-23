# Fuzzy Logic

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Classical (Crisp) Logic vs. Fuzzy Logic](#2-classical-crisp-logic-vs-fuzzy-logic)
3.  [Fuzzy Sets](#3-fuzzy-sets)
    * [Membership Function ($\mu$)](#31-membership-function-mu)
    * [Degree of Membership](#32-degree-of-membership)
4.  [Fuzzy Set Operations](#4-fuzzy-set-operations)
    * [Fuzzy Union (OR)](#41-fuzzy-union-or)
    * [Fuzzy Intersection (AND)](#42-fuzzy-intersection-and)
    * [Fuzzy Complement (NOT)](#43-fuzzy-complement-not)
5.  [Fuzzy Rules (IF-THEN Rules)](#5-fuzzy-rules-if-then-rules)
6.  [Fuzzy Inference System (FIS)](#6-fuzzy-inference-system-fis)
    * [6.1 Fuzzification](#61-fuzzification)
    * [6.2 Fuzzy Inference Engine](#62-fuzzy-inference-engine)
    * [6.3 Defuzzification](#63-defuzzification)
7.  [Advantages](#7-advantages)
8.  [Limitations](#8-limitations)
9.  [Applications](#9-applications)
10. [See Also](#10-see-also)

---

## 1. Introduction

Fuzzy Logic, introduced by Lotfi A. Zadeh in 1965, is a form of many-valued logic that deals with reasoning that is approximate rather than fixed and exact. Unlike classical Boolean logic, where statements are either entirely true or entirely false (0 or 1), fuzzy logic allows for degrees of truth, represented by values between 0 and 1. This enables it to handle the inherent imprecision and uncertainty present in real-world human reasoning and complex systems.

## 2. Classical (Crisp) Logic vs. Fuzzy Logic

| Feature             | Classical (Crisp) Logic                               | Fuzzy Logic                                              |
|---------------------|-------------------------------------------------------|----------------------------------------------------------|
| **Values** | Binary (True/1 or False/0)                            | Continuous (between 0 and 1)                             |
| **Membership** | Either entirely in a set or entirely out (binary)     | Gradual; an element can partially belong to a set        |
| **Decision-making** | Sharp boundaries, abrupt transitions                  | Smooth transitions, handles vagueness                     |
| **Example** | A person is either TALL or NOT TALL                   | A person can be "somewhat tall" or "very tall"           |

## 3. Fuzzy Sets

A **fuzzy set** is a set whose elements have degrees of membership. For a universal set $U$, a fuzzy set $A$ in $U$ is defined as a set of ordered pairs:

$$ A = \{ (x, \mu_A(x)) \mid x \in U \} $$

Where $\mu_A(x)$ is the **membership function** of element $x$ in fuzzy set $A$.

### 3.1. Membership Function ($\mu$)

The **membership function** defines the degree to which an element belongs to a fuzzy set. It maps each element $x$ in the universal set $U$ to a value in the interval $[0, 1]$:

$$ \mu_A: U \to [0, 1] $$

Common shapes for membership functions include triangular, trapezoidal, Gaussian, and sigmoidal functions. The choice of shape depends on the application and how the concept's fuzziness is best represented.

### 3.2. Degree of Membership

The value $\mu_A(x)$ represents the **degree of membership** of $x$ in fuzzy set $A$.
* $\mu_A(x) = 1$: $x$ is fully a member of set $A$.
* $\mu_A(x) = 0$: $x$ is not a member of set $A$.
* $0 < \mu_A(x) < 1$: $x$ is a partial member of set $A$.

For example, a fuzzy set "Tall" for height might have $\mu_{Tall}(1.70m) = 0.6$ (moderately tall) and $\mu_{Tall}(1.95m) = 0.95$ (very tall).

## 4. Fuzzy Set Operations

Fuzzy logic extends classical set operations (AND, OR, NOT) to fuzzy sets. The most common operators are based on minimum (min) and maximum (max) t-norms and t-conorms.

### 4.1. Fuzzy Union (OR)

The union of two fuzzy sets $A$ and $B$ (representing logical OR) is defined by taking the maximum of their membership degrees for each element:

$$ \mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x)) $$

### 4.2. Fuzzy Intersection (AND)

The intersection of two fuzzy sets $A$ and $B$ (representing logical AND) is defined by taking the minimum of their membership degrees for each element:

$$ \mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x)) $$

### 4.3. Fuzzy Complement (NOT)

The complement of a fuzzy set $A$ is defined as:

$$ \mu_{\bar{A}}(x) = 1 - \mu_A(x) $$

## 5. Fuzzy Rules (IF-THEN Rules)

Fuzzy logic systems are typically based on a set of IF-THEN rules, derived from expert knowledge or data. These rules connect fuzzy input concepts to fuzzy output concepts.

**General Form:**
`IF (antecedent is fuzzy_set_1) AND (antecedent is fuzzy_set_2) THEN (consequent is fuzzy_set_3)`

**Examples:**
* `IF temperature IS cold AND humidity IS high THEN fan_speed IS low`
* `IF food_quality IS excellent THEN tip_percentage IS high`

Each part of the rule (antecedent and consequent) refers to a fuzzy set defined by a membership function.

## 6. Fuzzy Inference System (FIS)

A Fuzzy Inference System (FIS), also known as a Fuzzy Logic Controller (FLC), is the core unit of a fuzzy logic system. It processes inputs, applies fuzzy rules, and produces an output. A typical FIS consists of three main components:

### 6.1 Fuzzification

* This is the process of converting crisp (numerical) input values into fuzzy values (degrees of membership to predefined fuzzy sets).
* For each crisp input, its degree of membership in various relevant fuzzy sets is determined using the corresponding membership functions.
    * Example: If temperature is 22°C, how much is it "cold", "moderate", or "warm"?

### 6.2 Fuzzy Inference Engine

* This component applies the fuzzy IF-THEN rules to the fuzzified inputs.
* It evaluates the antecedents (IF-parts) of all rules using fuzzy operators (AND, OR) to get a single degree of truth for each rule.
* Based on this degree of truth, it applies the rule to its consequent (THEN-part) to produce a fuzzy output set for each rule. This often involves "clipping" or "scaling" the consequent's membership function.
* The fuzzy outputs from all active rules are then combined (usually using fuzzy union/maximum operator) to form a single combined fuzzy output set.

### 6.3 Defuzzification

* This is the process of converting the aggregated fuzzy output set from the inference engine back into a single, crisp (numerical) output value.
* This is necessary because, for practical applications (e.g., controlling a fan speed), a crisp control signal is usually required.
* Common defuzzification methods include:
    * **Centroid (Center of Gravity - COG):** Calculates the center of the area under the aggregated membership function.
    * **Mean of Maximum (MOM):** Takes the average of the output values that have the maximum membership degree.
    * **Smallest of Maximum (SOM) / Largest of Maximum (LOM):** Takes the smallest/largest value that has the maximum membership degree.

## 7. Advantages

* **Handles Uncertainty and Imprecision:** Excellent for situations where human expert knowledge is vague or approximate.
* **Robustness:** Can be quite robust to noisy or incomplete data.
* **Intuitive and Interpretable:** Rules are often human-readable and understandable, making system behavior easier to interpret than black-box models.
* **Low Computational Cost:** Often simpler and faster to compute than complex mathematical models, once designed.
* **Rapid Prototyping:** Can be developed relatively quickly based on expert knowledge.

## 8. Limitations

* **No Automatic Learning:** Unlike neural networks, traditional fuzzy logic systems do not inherently learn from data (though hybrid systems exist). Rules and membership functions often need to be manually defined or tuned.
* **Expert Dependence:** Relies heavily on expert knowledge for defining rules and membership functions, which can be difficult to acquire and might be subjective.
* **Scalability:** The number of rules can grow exponentially with the number of input variables and fuzzy sets, leading to complex rule bases.
* **Validation:** Validating and verifying fuzzy systems can be challenging.

## 9. Applications

Fuzzy logic is widely used in control systems and decision-making due to its ability to mimic human-like reasoning:
* **Consumer Electronics:** Washing machines, air conditioners, cameras, camcorders (e.g., controlling wash cycles, fan speeds, image stabilization).
* **Industrial Control:** Process control, robotics, automotive systems (e.g., anti-lock braking systems, automatic transmissions).
* **Medical Systems:** Diagnosis, drug delivery.
* **Finance:** Stock market prediction, risk assessment.
* **Environmental Control:** Water quality management, smart energy grids.

## 10. See Also

* [Classical Sets and Logic](../mathematical_foundations/classical_logic.md) (Potential future topic folder)
* [Neural Networks (for Hybrid Systems)](../neural_networks/introduction.md) (Potential future topic folder for ANFIS)

---

See the Python implementation of a Fuzzy Inference System [here](fuzzy_logic.py).