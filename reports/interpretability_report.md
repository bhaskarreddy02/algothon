# PredictiveGuard: Phase 6 Model Interpretability & Explainability Report
**ALGOTHON'26 — Track: ALG-DATA-02 "Predict What Happens Next"**  
**Methodology:** TreeSHAP (Tree Explainer), Permutation Feature Importance (PR-AUC Drop), Native Split/Gain Metrics  
**Explaining Model:** Frozen `final_pipeline.joblib` (Tuned LightGBM with 11 Physics-Informed & Interaction Features)  
**Sample Cohort:** 2,000 Representative Development Observations  
**Attribution Space:** Raw-margin / Log-odds space (additive contributions before logistic sigmoid transformation)

---

## 1. Executive Interpretability Summary
- **Black-Box Elimination:** In mission-critical predictive maintenance, factory operators cannot act on an ungrounded probability score. We implemented a unified interpretability framework linking mathematical SHAP attributions directly to physical machine failure mechanisms.
- **Triangulated Importance Verification:**
  - Evaluated feature importance across three independent methodologies: **TreeSHAP mean absolute attribution**, **Permutation Importance (PR-AUC degradation)**, and **Native Tree Gain**.
  - Across these evaluations, **physics-informed features provide substantial predictive value alongside the original sensor telemetry**. In Permutation Importance, shuffling `low_speed_low_tempdiff` causes a **$0.2813$ drop in PR-AUC**, followed by `overstrain_ratio` (**$0.2150$ drop**) and `speed_torque_ratio` (**$0.1358$ drop**).
- **Attribution Space & Link Function:**
  - TreeSHAP explains the LightGBM classifier in its native **raw-margin (log-odds) space**.
  - Each SHAP value $\phi_i$ represents an additive contribution to the base log-odds value ($\mathbb{E}[f(x)] = -4.5975$).
  - The sum of base value and feature attributions forms the model logit $z = \mathbb{E}[f(x)] + \sum_i \phi_i$, which maps monotonically through the logistic sigmoid link $\sigma(z) = \frac{1}{1 + e^{-z}}$ into the final failure probability.
- **Non-Linear Dynamics Consistent with Machine Physics:**
  - `overstrain_ratio` exhibits a sharp threshold transition at $\text{ratio} = 1.0$, which is consistent with the physical failure mechanism of structural capacity limits.
  - `power_w` exhibits a dual-sided risk profile, remaining negative (protective) within the verified safe band ($3,500\text{ W} - 9,000\text{ W}$) and shifting positive when power drops below $3.5\text{ kW}$ or exceeds $9.0\text{ kW}$.
  - `low_speed_low_tempdiff` isolates the thermal convection envelope ($\Delta T < 8.6\text{ K}$, $\omega < 1,380\text{ rpm}$), delivering a $+5.21$ log-odds risk attribution for heat dissipation failures.
- **Decision-Support Framing:** Model explanations provide decision support rather than autonomous maintenance prescriptions. Recommendations are designed to direct technician inspections toward high-probability root causes.

---

## 2. Triangulated Feature Importance Comparison

To prevent reliance on a single metric artifact, we compared feature significance across three independent paradigms:
1. **TreeSHAP $\text{Mean } |\text{SHAP}|$:** Average absolute impact on failure log-odds across 2,000 samples.
2. **Permutation Importance ($\Delta\text{PR-AUC}$):** Empirical drop in Precision-Recall AUC when the feature column is randomly shuffled ($n=5$ repeats).
3. **Native Tree Gain:** Total training loss reduction contributed by splits on the feature.

| Rank | Feature Name | Feature Category | Mean $|\text{SHAP}|$ (Log-odds) | Permutation Drop ($\Delta\text{PR-AUC}$) | Native Split Gain | Primary Physical Mechanism Governed |
|---|---|---|---|---|---|---|
| **1** | `Tool wear [min]` | Base Telemetry | **$0.3780$** | $0.1269 \pm 0.0060$ | $26,052$ | Cumulative cutting duration; baseline wearout envelope |
| **2** | `Rotational speed [rpm]` | Base Telemetry | **$0.2503$** | $0.0337 \pm 0.0045$ | $18,466$ | Spindle kinematics; convection airflow driving heat dissipation |
| **3** | `power_w` | Direct Physical Law | **$0.2214$** | $0.0235 \pm 0.0071$ | $3,730$ | Mechanical power draw $P = \tau \cdot \frac{2\pi \omega}{60}$ |
| **4** | `speed_torque_ratio` | Kinematic Interaction | **$0.2187$** | $0.1358 \pm 0.0152$ | **$34,432$** | Mechanical load impedance; separates low-torque idling from heavy cutting |
| **5** | `overstrain_ratio` | Direct Physical Law | **$0.1310$** | **$0.2150 \pm 0.0259$** | $26,026$ | Product of wear and torque normalized by variant capacity |
| **6** | `low_speed_low_tempdiff` | Direct Physical Law | **$0.1207$** | **$0.2813 \pm 0.0071$** | $23,521$ | Thermal entrapment severity when $\Delta T < 8.6\text{ K}$ and $\omega < 1,380\text{ rpm}$ |
| **7** | `Air temperature [K]` | Base Telemetry | $0.0770$ | $0.0063 \pm 0.0004$ | $869$ | Ambient environmental baseline |
| **8** | `Torque [Nm]` | Base Telemetry | $0.0710$ | $0.0113 \pm 0.0036$ | $2,539$ | Cutting tool mechanical resistance |
| **9** | `tool_wear_in_critical_band` | Direct Physical Law | $0.0667$ | $0.0005 \pm 0.0015$ | $4,931$ | Binary indicator for tool replacement window $[200, 240]\text{ min}$ |
| **10** | `wear_torque` | Kinematic Interaction | $0.0638$ | $-0.0001 \pm 0.0016$ | $1,893$ | Raw cumulative mechanical stress accumulation |
| **11** | `temp_ratio` | Kinematic Interaction | $0.0630$ | $0.0084 \pm 0.0009$ | $1,419$ | Relative thermodynamic heat accumulation |
| **12** | `Process temperature [K]` | Base Telemetry | $0.0461$ | $0.0089 \pm 0.0022$ | $1,437$ | Machining tool heat generation |
| **13** | `power_distance_to_safe_band` | Direct Physical Law | $0.0383$ | $0.0105 \pm 0.0027$ | $27,438$ | Metric distance outside safe power envelope $[3.5\text{ kW}, 9.0\text{ kW}]$ |
| **14** | `power_above_limit` | Direct Physical Law | $0.0280$ | $0.0175 \pm 0.0052$ | $15,341$ | Excess power draw indicator ($>9,000\text{ W}$) |
| **15** | `power_below_limit` | Direct Physical Law | $0.0207$ | $0.0185 \pm 0.0010$ | $10,478$ | Insufficient power draw indicator ($<3,500\text{ W}$) |
| **16** | `temp_diff` | Direct Physical Law | $0.0196$ | $0.0039 \pm 0.0004$ | $1,293$ | Temperature gradient driving convective heat transfer |
| **17** | `tool_wear_critical_proximity`| Direct Physical Law | $0.0183$ | $-0.0009 \pm 0.0008$ | $160$ | Exponential proximity metric to failure wear boundary |
| **18** | `Type_H` | Categorical Encoding | $0.0176$ | $0.0000 \pm 0.0000$ | $156$ | High product variant indicator |
| **19** | `Type_L` | Categorical Encoding | $0.0087$ | $0.0011 \pm 0.0011$ | $109$ | Low product variant indicator |
| **20** | `Type_M` | Categorical Encoding | $0.0058$ | $0.0001 \pm 0.0001$ | $21$ | Medium product variant indicator |
| **21** | `is_known_type` | Integrity Safeguard | $0.0000$ | $0.0000 \pm 0.0000$ | $0$ | Fallback flag for unseen categorical types |

*Takeaway:* The permutation test confirms that `low_speed_low_tempdiff` and `overstrain_ratio` provide substantial predictive value alongside the original sensor telemetry, with shuffling either feature causing over $0.21$ drop in PR-AUC.

---

## 3. Global Attributions: SHAP Summary & Directionality (Figure 16)

The SHAP summary beeswarm plot ([`reports/figures/16_shap_summary_beeswarm.png`](file:///c:/Users/patha/OneDrive/Desktop/algothon/reports/figures/16_shap_summary_beeswarm.png)) visualizes feature magnitude and directionality across 2,000 machines in log-odds space:
- **`Tool wear [min]`:** High wear durations (red points) contribute positively to failure log-odds ($+1.0$ to $+3.5$), while low wear durations (blue points) provide a negative log-odds baseline ($-0.3$ to $-0.5$).
- **`power_w`:** Shows bidirectional risk contributions: extreme high power (red) and extreme low power (blue) increase failure log-odds, while intermediate power values ($5,000-7,000\text{ W}$) reduce failure log-odds.
- **`overstrain_ratio`:** Telemetry below the capacity limit ($0.8$) contributes near-zero or negative log-odds, whereas values exceeding $1.0$ contribute strong positive log-odds ($+2.0$ to $+5.0$).
- **`low_speed_low_tempdiff`:** Remains zero for non-risk samples, but produces large positive log-odds shifts ($+4.0$ to $+6.0$) when thermal deficit conditions occur.

---

## 4. Physical Interaction & Dependence Analysis (Figure 17)

Figure 17 ([`reports/figures/17_shap_dependence_interactions.png`](file:///c:/Users/patha/OneDrive/Desktop/algothon/reports/figures/17_shap_dependence_interactions.png)) demonstrates feature attributions across operational regimes:

1. **Overstrain Ratio vs. Torque (Subplot 0,0):**
   - The log-odds SHAP attribution remains flat near $0.0$ for ratios below $0.9$.
   - As the ratio crosses $1.0$ (consistent with the physical mechanism where cutting stress exceeds material structural limits), the attribution jumps from $0.0$ to $+4.5$ log-odds.
   - Colored by torque: observations with higher torque reach critical overstrain at lower tool wear durations.
2. **Thermal Convection Entrapment (Subplot 0,1):**
   - In nominal cooling conditions, the interaction index is zero and contributes negative log-odds.
   - As the combined deficit index increases, SHAP attributions rise monotonically toward $+5.5$ log-odds, consistent with the physical heat dissipation failure mechanism.
3. **Mechanical Power Envelope (Subplot 1,0):**
   - Within the safe operational range ($3,500\text{ W} \le P \le 9,000\text{ W}$), the attribution is negative ($-0.1$ to $-0.3$ log-odds).
   - Outside this band ($P < 3,500\text{ W}$ or $P > 9,000\text{ W}$), attributions shift positive up to $+2.5$ log-odds, reflecting spindle overload or underload regimes.
4. **Tool Wear Duration (Subplot 1,1):**
   - At wear durations below $150\text{ min}$, wear contributes negative log-odds.
   - Above $200\text{ min}$ (consistent with the tool replacement threshold), attributions become strongly positive, modulated by concurrent mechanical torque.

*Note on Causality:* SHAP attributions reflect statistical model contributions that align with known physical failure laws; they quantify model sensitivity rather than establishing independent physical causality.

---

## 5. Local Explainability Case Studies (Figure 18)

Four representative operational scenarios were evaluated to demonstrate local decision-support utility. Recommendations serve as diagnostic decision support for maintenance personnel rather than autonomous prescriptions.

### Case Study 1: Heat Dissipation Failure (HDF, Sample 25)
- **Operational Readings:** Type L, Air Temp = $302.4\text{ K}$, Process Temp = $310.2\text{ K}$, Speed = $1,351\text{ rpm}$, Torque = $45.1\text{ Nm}$, Wear = $168\text{ min}$.
- **Physical Analysis:** Temperature gradient $\Delta T = 7.8\text{ K}$ (below $8.6\text{ K}$) and speed is $1,351\text{ rpm}$ (below $1,380\text{ rpm}$).
- **Predicted Probability:** **$95.3\%$** (Base log-odds: $-4.60$, Final log-odds: $+3.01$)
- **Top SHAP Risk Drivers (Log-odds):**
  1. `low_speed_low_tempdiff`: **$+5.211$**
  2. `Rotational speed [rpm]`: **$+1.252$**
  3. `speed_torque_ratio`: **$+0.477$**
- **Decision-Support Recommendation:**
  > *"WARNING (Decision Support): Operating telemetry indicates potential convective cooling deficit. Inspect coolant flow, radiator fan, and clean thermal heat sink; corrective cooling measures may be needed."*

---

### Case Study 2: Mechanical Power Failure (PWF, Sample 34)
- **Operational Readings:** Type M, Air Temp = $298.9\text{ K}$, Process Temp = $308.2\text{ K}$, Speed = $1,372\text{ rpm}$, Torque = $63.6\text{ Nm}$, Wear = $30\text{ min}$.
- **Physical Analysis:** Mechanical power $P = 63.6 \times \frac{2\pi \cdot 1,372}{60} = 9,137.8\text{ W}$, exceeding the $9,000\text{ W}$ design threshold by $+137.8\text{ W}$.
- **Predicted Probability:** **$96.9\%$** (Base log-odds: $-4.60$, Final log-odds: $+3.46$)
- **Top SHAP Risk Drivers (Log-odds):**
  1. `power_above_limit`: **$+2.782$**
  2. `power_distance_to_safe_band`: **$+2.400$**
  3. `speed_torque_ratio`: **$+2.106$**
- **Decision-Support Recommendation:**
  > *"WARNING (Decision Support): Mechanical power draw is outside the safe operating band (3.5kW - 9.0kW). Check spindle motor drive, inverter, and mechanical transmission; feed rate adjustments may be required."*

---

### Case Study 3: Overstrain Failure (OSF, Sample 35)
- **Operational Readings:** Type L, Air Temp = $298.7\text{ K}$, Process Temp = $309.8\text{ K}$, Speed = $1,354\text{ rpm}$, Torque = $53.3\text{ Nm}$, Wear = $212\text{ min}$.
- **Physical Analysis:** Stress product $\text{wear} \times \text{torque} = 212 \times 53.3 = 11,299.6\text{ min}\cdot\text{Nm}$, exceeding the Type L threshold of $11,000\text{ min}\cdot\text{Nm}$ ($\text{ratio} = 1.027$).
- **Predicted Probability:** **$96.6\%$** (Base log-odds: $-4.60$, Final log-odds: $+3.35$)
- **Top SHAP Risk Drivers (Log-odds):**
  1. `overstrain_ratio`: **$+4.340$**
  2. `Tool wear [min]`: **$+1.499$**
  3. `speed_torque_ratio`: **$+1.145$**
- **Decision-Support Recommendation:**
  > *"CRITICAL (Decision Support): Operating telemetry is consistent with tool overstrain. Inspect cutting insert and tool holder; replacement may be required."*

---

### Case Study 4: Nominal Machine Operation (Sample 0)
- **Operational Readings:** Type M, Air Temp = $302.0\text{ K}$, Process Temp = $310.9\text{ K}$, Speed = $1,456\text{ rpm}$, Torque = $47.2\text{ Nm}$, Wear = $54\text{ min}$.
- **Physical Analysis:** Power in safe band ($7,196.7\text{ W}$), low wear duration ($54\text{ min}$), adequate thermal dissipation ($\Delta T = 8.9\text{ K}$, $\omega = 1,456\text{ rpm}$).
- **Predicted Probability:** **$0.6\%$** (Base log-odds: $-4.60$, Final log-odds: $-5.06$)
- **Top SHAP Mitigating Factors (Log-odds):**
  1. `Tool wear [min]` ($54\text{ min}$): **$-0.272$**
  2. `Rotational speed [rpm]` ($1,456\text{ rpm}$): **$-0.132$**
  3. `power_w` ($7,196.7\text{ W}$): **$-0.129$**
- **Decision-Support Recommendation:**
  > *"NOMINAL (Decision Support): Machine operating within verified safe physical envelopes. Continue standard monitoring."*

---

## 6. Actionable Decision Framework for Maintenance Teams

The following decision-support guidelines assist maintenance technicians in prioritizing diagnostic inspections:

| Dominant Risk Signal | Trigger Condition | Recommended Technician Inspection Protocol |
|---|---|---|
| **Structural Overstrain** | `overstrain_ratio > 0.95` or `wear_torque > 10,500` | Inspect cutting tool insert and tool holder for wear or deformation; replacement may be needed. Check workpiece material hardness. |
| **Thermal Convection Deficit** | `low_speed_low_tempdiff > 0.0` ($\Delta T < 8.6\text{ K}$ & $\omega < 1,380$) | Inspect coolant line pressure and delivery nozzles. Check radiator fan and clean air filters. |
| **High Spindle Power** | `power_w > 9,000 W` or `power_above_limit > 0` | Review feed rate and depth of cut. Inspect spindle bearings and transmission lubrication for binding or mechanical resistance. |
| **Low Spindle Power** | `power_w < 3,500 W` or `power_below_limit > 0` | Check for spindle belt slippage, motor drive inverter fault, or unexpected tool disengagement. |
| **End-of-Life Tool Wear** | `Tool wear > 200 min` | Plan scheduled tool inspection and possible swap during upcoming maintenance transition. |

---

## 7. Artifact Manifest
- **Interpretability Module:** [`src/explain.py`](file:///c:/Users/patha/OneDrive/Desktop/algothon/src/explain.py)
- **Feature Importance Comparison:** [`reports/feature_importance_comparison.csv`](file:///c:/Users/patha/OneDrive/Desktop/algothon/reports/feature_importance_comparison.csv)
- **Interpretability Summary JSON:** [`reports/interpretability_summary.json`](file:///c:/Users/patha/OneDrive/Desktop/algothon/reports/interpretability_summary.json)
- **Figure 16:** [`reports/figures/16_shap_summary_beeswarm.png`](file:///c:/Users/patha/OneDrive/Desktop/algothon/reports/figures/16_shap_summary_beeswarm.png)
- **Figure 17:** [`reports/figures/17_shap_dependence_interactions.png`](file:///c:/Users/patha/OneDrive/Desktop/algothon/reports/figures/17_shap_dependence_interactions.png)
- **Figure 18:** [`reports/figures/18_shap_waterfall_case_studies.png`](file:///c:/Users/patha/OneDrive/Desktop/algothon/reports/figures/18_shap_waterfall_case_studies.png)
- **Automated Tests:** [`tests/test_explain.py`](file:///c:/Users/patha/OneDrive/Desktop/algothon/tests/test_explain.py) (**34 / 34 tests passing**)
