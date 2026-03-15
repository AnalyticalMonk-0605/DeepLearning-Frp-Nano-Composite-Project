import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("frp_rnn_model_results.xlsx", sheet_name="Full_Test_Predictions")
sns.set_theme(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(8,6))
sns.lineplot(x="Nano_Silica_%", y="Actual_Tensile_Stress", data=df, label="Actual", ax=ax1, color="steelblue", linewidth=2.5)
sns.lineplot(x="Nano_Silica_%", y="Predicted_Tensile_Stress", data=df, label="Predicted", ax=ax1, color="orange", linewidth=2.5)

ax1.set_xlabel("Nano Silica (%)", fontsize=12)
ax1.set_ylabel("Tensile Stress (MPa)", fontsize=12)
ax1.set_title("Actual vs Predicted Tensile Stress", fontsize=14, weight='bold')
ax1.legend(frameon=True, loc='best', fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,6))
sns.set_style("white")
sns.regplot(x="Actual_Tensile_Stress", y="Predicted_Tensile_Stress", data=df,
            scatter_kws={"color": "teal", "alpha": 0.6}, line_kws={"color": "red", "lw":2})
plt.title("Regression Fit: Actual vs Predicted Tensile Stress", fontsize=14, weight='bold')
plt.xlabel("Actual Tensile Stress (MPa)", fontsize=12)
plt.ylabel("Predicted Tensile Stress (MPa)", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

metrics = pd.read_excel("frp_rnn_model_results.xlsx", sheet_name="Evaluation_Metrics")
plt.figure(figsize=(7,4))
sns.barplot(y="Metric", x="Value", data=metrics, palette="crest")
plt.title("Model Evaluation Summary", fontsize=14, weight='bold')
plt.xlabel("Metric Value", fontsize=12)
plt.ylabel("")
plt.tight_layout()
plt.show()

df["Residuals_Tensile"] = df["Actual_Tensile_Stress"] - df["Predicted_Tensile_Stress"]
plt.figure(figsize=(8,5))
sns.histplot(df["Residuals_Tensile"], kde=True, color="midnightblue")
plt.title("Residual Distribution: Tensile Stress", fontsize=14, weight='bold')
plt.xlabel("Residual (Actual - Predicted)", fontsize=12)
plt.tight_layout()
plt.show()

opt = pd.read_excel("frp_rnn_model_results.xlsx", sheet_name="Optimal_Results")
plt.figure(figsize=(8,6))
sns.scatterplot(x="Nano_Silica_%", y="Predicted_Tensile_Stress", hue="Predicted_Flexural_Stress",
                size="Predicted_Flexural_Stress", data=opt, palette="viridis", legend="brief")
plt.title("Predicted Mechanical Properties at Optimal Compositions", fontsize=14, weight='bold')
plt.xlabel("Nano Silica (%)", fontsize=12)
plt.ylabel("Predicted Tensile Stress (MPa)", fontsize=12)
plt.tight_layout()
plt.show()

import plotly.express as px

fig = px.scatter(df, x="Actual_Tensile_Stress", y="Predicted_Tensile_Stress",
                 color="Nano_Silica_%", size="Predicted_Flexural_Stress",
                 title="Interactive Visualization: Model Predictions",
                 template="plotly_white")
fig.update_layout(title_font=dict(size=18, family="Arial", color="black"))
fig.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("frp_rnn_model_results.xlsx", sheet_name="Full_Test_Predictions")
sns.set_theme(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(8,6))
sns.lineplot(x="Nano_Silica_%", y="Actual_Tensile_Stress", data=df, label="Actual", ax=ax1, color="steelblue", linewidth=2.5)
sns.lineplot(x="Nano_Silica_%", y="Predicted_Tensile_Stress", data=df, label="Predicted", ax=ax1, color="orange", linewidth=2.5)

ax1.set_xlabel("Nano Silica (%)", fontsize=12)
ax1.set_ylabel("Tensile Stress (MPa)", fontsize=12)
ax1.set_title("Actual vs Predicted Tensile Stress", fontsize=14, weight='bold')
ax1.legend(frameon=True, loc='best', fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,6))
sns.set_style("white")
sns.regplot(x="Actual_Tensile_Stress", y="Predicted_Tensile_Stress", data=df,
            scatter_kws={"color": "teal", "alpha": 0.6}, line_kws={"color": "red", "lw":2})
plt.title("Regression Fit: Actual vs Predicted Tensile Stress", fontsize=14, weight='bold')
plt.xlabel("Actual Tensile Stress (MPa)", fontsize=12)
plt.ylabel("Predicted Tensile Stress (MPa)", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

metrics = pd.read_excel("frp_rnn_model_results.xlsx", sheet_name="Evaluation_Metrics")
plt.figure(figsize=(7,4))
sns.barplot(y="Metric", x="Value", data=metrics, palette="crest")
plt.title("Model Evaluation Summary", fontsize=14, weight='bold')
plt.xlabel("Metric Value", fontsize=12)
plt.ylabel("")
plt.tight_layout()
plt.show()

df["Residuals_Tensile"] = df["Actual_Tensile_Stress"] - df["Predicted_Tensile_Stress"]
plt.figure(figsize=(8,5))
sns.histplot(df["Residuals_Tensile"], kde=True, color="midnightblue")
plt.title("Residual Distribution: Tensile Stress", fontsize=14, weight='bold')
plt.xlabel("Residual (Actual - Predicted)", fontsize=12)
plt.tight_layout()
plt.show()

opt = pd.read_excel("frp_rnn_model_results.xlsx", sheet_name="Optimal_Results")
plt.figure(figsize=(8,6))
sns.scatterplot(x="Nano_Silica_%", y="Predicted_Tensile_Stress", hue="Predicted_Flexural_Stress",
                size="Predicted_Flexural_Stress", data=opt, palette="viridis", legend="brief")
plt.title("Predicted Mechanical Properties at Optimal Compositions", fontsize=14, weight='bold')
plt.xlabel("Nano Silica (%)", fontsize=12)
plt.ylabel("Predicted Tensile Stress (MPa)", fontsize=12)
plt.tight_layout()
plt.show()

import plotly.express as px
fig = px.scatter(df, x="Actual_Tensile_Stress", y="Predicted_Tensile_Stress",
                 color="Nano_Silica_%", size="Predicted_Flexural_Stress",
                 title="Interactive Visualization: Model Predictions",
                 template="plotly_white")
fig.update_layout(title_font=dict(size=18, family="Arial", color="black"))
fig.show()