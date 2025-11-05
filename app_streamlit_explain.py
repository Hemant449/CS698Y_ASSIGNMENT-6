
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Optional: SHAP (falls back gracefully if not installed)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Student Outcome: Fairness + Explainability (A6)", layout="wide")
st.title("🎓 Student Outcome Prediction — Fairness + Explainability (Assignment‑6)")
st.caption("Extended from Assignment‑4 with **global** and **local** explanations, what‑if analysis, and export.")

# -----------------------------
# Helpers
# -----------------------------
APP_VERSION = "v6.0-a6"
DEFAULT_TARGET_MAPPING = {'Graduate': 2, 'Enrolled': 1, 'Dropout': 0}

PROTECTED_ATTRIBUTES_DEFAULT = [
    'Marital status',
    'Nacionality',
    'Gender',
    'Scholarship holder',
    'Displaced',
    'Debtor',
    'Age at enrollment'  # used to derive AgeBin
]

def age_to_bin(age):
    try:
        age = float(age)
    except Exception:
        return np.nan
    if age <= 20:
        return '<=20'
    elif 21 <= age <= 25:
        return '21-25'
    elif 26 <= age <= 30:
        return '26-30'
    else:
        return '>30'

@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str):
    return pd.read_csv(path, sep=None, engine="python")  # auto-detect sep

def compute_bias_table(df: pd.DataFrame, attr: str, target_col="Target"):
    tbl = pd.DataFrame(index=df[attr].value_counts().index)
    tbl['count'] = df[attr].value_counts()
    tbl['representation'] = df[attr].value_counts(normalize=True)
    dropout_mask = df[target_col] == 'Dropout'
    dr = df[dropout_mask][attr].value_counts() / df[attr].value_counts()
    tbl['Dropout_rate'] = dr.fillna(0.0)
    return tbl.sort_values('count', ascending=False)

def disparate_impact_and_dpd(labels_pred: pd.Series, groups: pd.Series, favorable_label: str):
    rates = labels_pred.groupby(groups).apply(lambda x: (x == favorable_label).mean()).sort_index()
    if (rates.max() or 0) == 0:
        di = None
        dpd = None
    else:
        base_group = rates.idxmax()
        base_rate = rates.loc[base_group]
        di = rates / base_rate
        dpd = rates - base_rate
    return rates, di, dpd

def equal_opportunity(true_labels: pd.Series, pred_labels: pd.Series, groups: pd.Series, mapping: dict, favorable_label='Graduate'):
    fv = mapping[favorable_label]
    mask = true_labels == favorable_label
    tl = true_labels[mask].map(mapping)
    pl = pred_labels[mask].map(mapping)
    gp = groups[mask]
    if tl.empty:
        return None, None
    def tpr_of_group(idx):
        gmask = (gp == idx)
        if gmask.sum() == 0:
            return 0.0
        y_true = tl[gmask]
        y_pred = pl[gmask]
        if (y_true == fv).sum() == 0:
            return 0.0
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        tpr = cm[fv, fv] / (y_true == fv).sum()
        return float(tpr)
    uniq = gp.dropna().unique()
    tprs = pd.Series({g: tpr_of_group(g) for g in uniq})
    if tprs.empty:
        return None, None
    base = tprs.max()
    eo_diff = tprs - base
    return tprs, eo_diff

def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    names = []
    # numeric first
    names.extend(numeric_cols)
    # then one-hot names
    try:
        ohe = preprocessor.named_transformers_["cat"]
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
    except Exception:
        ohe_names = []
    names.extend(ohe_names)
    return names

def aggregate_importances_to_base(importances, feat_names):
    # Aggregate one-hot importances back to their base column (sum)
    agg = {}
    for imp, nm in zip(importances, feat_names):
        if "cat__" in nm:
            base = nm.split("cat__")[1]
            # base like "Gender_M" or "Gender_F" -> split by first underscore
            if "_" in base:
                base = base.split("_")[0]
        else:
            base = nm
        agg[base] = agg.get(base, 0.0) + float(imp)
    return pd.Series(agg).sort_values(ascending=False)

# -----------------------------
# Sidebar: links & data input
# -----------------------------
with st.sidebar:
    st.header("🔗 Links")
    repo_url = st.text_input("Code repository URL", value="https://github.com/<your-id>/student-outcome-fairness-a6")
    hosted_url = st.text_input("Hosted app URL", value="https://<streamlit-share-link>")
    st.write(f"**App version:** {APP_VERSION}")

    st.header("📥 Data")
    src = st.radio("Choose data source", ["Upload CSV", "Use local file (data.csv)"], index=0)
    if src == "Upload CSV":
        up = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    else:
        up = None
        st.info("Will attempt to read `data.csv` from the working directory.")

    st.divider()
    st.header("⚙️ Settings")
    rand_state = st.number_input("Random seed", value=42, step=1)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    favorable_label = st.selectbox("Favorable outcome label", ["Graduate", "Enrolled", "Dropout"], index=0)

    st.divider()
    st.header("🧷 Columns")
    st.caption("If your column names differ, edit them below to match your file.")
    target_col = st.text_input("Target column", value="Target")
    age_col = st.text_input("Age column used to make AgeBin", value="Age at enrollment")

# -----------------------------
# Load data
# -----------------------------
df = None
error = None
try:
    if up is not None:
        df = load_data_from_upload(up)
    else:
        if os.path.exists("data.csv"):
            df = load_data_from_path("data.csv")
        else:
            st.warning("No file uploaded and `data.csv` not found. Please upload a CSV.")
except Exception as e:
    error = str(e)

if error:
    st.error(f"Failed to load data: {error}")
    st.stop()

if df is None:
    st.stop()

st.success(f"Loaded data with shape {df.shape}")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found. Please fix the column name in the sidebar.")
    st.stop()

# Create AgeBin if possible
if age_col in df.columns:
    df['AgeBin'] = df[age_col].apply(age_to_bin)
else:
    st.warning(f"Age column '{age_col}' not found; fairness analysis for AgeBin will be skipped.")

# Ensure target mapping
target_vals = sorted(df[target_col].dropna().unique().tolist())
target_mapping = DEFAULT_TARGET_MAPPING.copy()
if set(target_vals) != set(target_mapping.keys()):
    target_mapping = {str(label): i for i, label in enumerate(target_vals)}
inverse_target_mapping = {v: k for k, v in target_mapping.items()}

# -----------------------------
# Descriptive bias
# -----------------------------
st.subheader("📊 Bias (Descriptive) by Protected Attributes")
prot_opts = [c for c in PROTECTED_ATTRIBUTES_DEFAULT if c in df.columns]
if 'AgeBin' in df.columns and 'AgeBin' not in prot_opts:
    prot_display = prot_opts + ['AgeBin']
else:
    prot_display = prot_opts

if not prot_display:
    st.info("No protected-attribute columns found. You can still train a model from the next section.")
else:
    choice = st.multiselect("Choose attributes to summarize", prot_display, default=prot_display)
    for attr in choice:
        st.markdown(f"**Subgroup report for `{attr}`**")
        try:
            tbl = compute_bias_table(df, attr, target_col=target_col)
            st.dataframe(tbl.style.format({'representation': '{:.4f}', 'Dropout_rate': '{:.4f}'}), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute subgroup table for '{attr}': {e}")

# -----------------------------
# Train/test split & preprocessing
# -----------------------------
st.subheader("🛠️ Train Model")
with st.status("Preparing features...", expanded=False) as status:
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).map(target_mapping)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if 'AgeBin' in X.columns and age_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != age_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder='drop'
    )

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand_state, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train_orig)
    X_test  = preprocessor.transform(X_test_orig)

    status.update(label="Features ready!", state="complete")

# -----------------------------
# Train baseline model
# -----------------------------
with st.status("Training baseline RandomForest...", expanded=False) as status:
    baseline = RandomForestClassifier(random_state=rand_state)
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    status.update(label="Baseline trained", state="complete")

acc_baseline = accuracy_score(y_test, y_pred)
st.metric("Baseline Accuracy", f"{acc_baseline:.4f}")
cr_baseline = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
with st.expander("Classification report (baseline)"):
    st.json(cr_baseline)

# Prepare fairness DF
y_test_cat = pd.Series(y_test, index=X_test_orig.index).map(inverse_target_mapping)
y_pred_cat = pd.Series(y_pred, index=X_test_orig.index).map(inverse_target_mapping)

# -----------------------------
# Fairness (baseline)
# -----------------------------
st.subheader("⚖️ Fairness (Baseline)")
prot_for_eval = []
for c in ['Gender', 'Marital status', 'AgeBin', 'Scholarship holder', 'Displaced', 'Debtor']:
    if c in X_test_orig.columns:
        prot_for_eval.append(c)

if not prot_for_eval:
    st.info("No protected attributes available in the test set to compute fairness metrics.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Disparate Impact / Demographic Parity Difference** (using predicted favorable rate)")
    with col2:
        st.caption("**Equal Opportunity Difference** (TPR among true favorable)")

    for attr in prot_for_eval:
        st.markdown(f"**Attribute:** `{attr}`")
        groups = X_test_orig[attr]
        rates, di, dpd = disparate_impact_and_dpd(y_pred_cat, groups, favorable_label=favorable_label)
        if rates is not None:
            c1, c2 = st.columns(2)
            with c1:
                df_rates = pd.DataFrame({"pred_favorable_rate": rates}).sort_index()
                if di is not None:
                    df_rates["disparate_impact_vs_best"] = di
                    df_rates["dpd_vs_best"] = dpd
                st.dataframe(df_rates.style.format("{:.3f}"), use_container_width=True)
        tprs, eo = equal_opportunity(y_test_cat, y_pred_cat, groups, target_mapping, favorable_label=favorable_label)
        if tprs is not None:
            c1, c2 = st.columns(2)
            with c2:
                df_eo = pd.DataFrame({"TPR": tprs, "EO_diff_vs_best": eo}).sort_index()
                st.dataframe(df_eo.style.format("{:.3f}"), use_container_width=True)

# -----------------------------
# 🔎 Global explanations
# -----------------------------
st.subheader("🧭 Global Explanations")
feat_names_full = get_feature_names(preprocessor, numeric_cols, categorical_cols)

# (1) Tree-based importance aggregated to base columns
importances = getattr(baseline, "feature_importances_", None)
if importances is not None and len(feat_names_full) == len(importances):
    agg_imp = aggregate_importances_to_base(importances, feat_names_full)
    st.markdown("**RandomForest feature importance (aggregated to base columns)**")
    st.dataframe(agg_imp.to_frame("importance").style.format("{:.6f}"))
else:
    st.info("Feature importances not available.")

# (2) Permutation importance on held-out set (encoded)
with st.status("Computing permutation importance (global)...", expanded=False):
    try:
        perm = permutation_importance(baseline, X_test, y_test, n_repeats=5, random_state=rand_state, n_jobs=-1)
        perm_s = pd.Series(perm.importances_mean, index=feat_names_full).sort_values(ascending=False)
        perm_agg = aggregate_importances_to_base(perm_s.values, perm_s.index.tolist())
        st.markdown("**Permutation importance (held‑out, aggregated)**")
        st.dataframe(perm_agg.to_frame("mean_importance").style.format("{:.6f}"))
    except Exception as e:
        st.warning(f"Permutation importance failed: {e}")

# (3) Partial Dependence for a chosen numeric feature
num_for_pdp = [c for c in numeric_cols if c in X_train_orig.columns]
if len(num_for_pdp) > 0:
    sel_pdp = st.selectbox("Choose a numeric feature for Partial Dependence", num_for_pdp, index=0)
    try:
        fig, ax = plt.subplots(figsize=(5,3))
        PartialDependenceDisplay.from_estimator(baseline, preprocessor.transform(X_train_orig), [feat_names_full.index(sel_pdp)], feature_names=feat_names_full, ax=ax)
        st.pyplot(fig, clear_figure=True, use_container_width=False)
    except Exception:
        # fallback: compute PD by sampling across quantiles
        grid = np.linspace(X_train_orig[sel_pdp].quantile(0.05), X_train_orig[sel_pdp].quantile(0.95), 20)
        samples = X_test_orig.copy().iloc[:200].copy()
        proba = []
        for val in grid:
            samples[sel_pdp] = val
            Xs = preprocessor.transform(samples)
            # average probability of favorable class
            if hasattr(baseline, "predict_proba"):
                fav_id = target_mapping.get(favorable_label, max(target_mapping.values()))
                p = baseline.predict_proba(Xs)[:, fav_id].mean()
            else:
                p = (baseline.predict(Xs) == target_mapping.get(favorable_label, 1)).mean()
            proba.append(p)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(grid, proba)
        ax.set_xlabel(sel_pdp); ax.set_ylabel(f"P({favorable_label})")
        ax.set_title("Partial dependence (fallback)")
        st.pyplot(fig, clear_figure=True, use_container_width=False)
else:
    st.info("No numeric features available for PDP.")

# -----------------------------
# 🧩 Local explanations (instance-level)
# -----------------------------
st.subheader("🔍 Local Explanations (pick a test row)")
if len(X_test_orig) > 0:
    row_idx = st.number_input("Row index from test set", min_value=int(X_test_orig.index.min()), max_value=int(X_test_orig.index.max()), value=int(X_test_orig.index.min()), step=1)
    if row_idx not in X_test_orig.index:
        st.warning("Chosen index not in test set.")
    else:
        instance_raw = X_test_orig.loc[[row_idx]]
        st.write("**Features (raw):**")
        st.dataframe(instance_raw.T, use_container_width=True)

        X_enc = preprocessor.transform(instance_raw)
        pred = baseline.predict(X_enc)[0]
        pred_label = inverse_target_mapping[pred]
        proba = None
        if hasattr(baseline, "predict_proba"):
            fav_id = target_mapping.get(pred_label, pred)
            proba = float(baseline.predict_proba(X_enc)[0][pred])
        st.metric("Predicted label", pred_label)
        if proba is not None:
            st.caption(f"Predicted class probability (for '{pred_label}'): **{proba:.3f}**")

        # SHAP (encoded)
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(baseline)
                shap_vals = explainer.shap_values(X_enc)
                # shap_values for multiclass: list of arrays [n_classes][n_rows,n_features]
                if isinstance(shap_vals, list):
                    # use predicted class for local bar plot
                    sv = shap_vals[pred][0]
                else:
                    sv = shap_vals[0]
                # top contributions
                contrib = pd.Series(sv, index=feat_names_full).sort_values(key=lambda x: np.abs(x), ascending=False)
                st.markdown("**Top contributing encoded features (SHAP, local)**")
                st.dataframe(contrib.head(20).to_frame("shap_value"))
                st.caption("Note: Encoded features are aggregated one‑hot columns.")

                # Global summary (optional)
                with st.expander("SHAP summary plot (subset)", expanded=False):
                    bg = X_train[:500] if X_train.shape[0] > 500 else X_train
                    shap.summary_plot(shap_vals if isinstance(shap_vals, list) else shap_vals, X_enc, feature_names=feat_names_full, show=False)
                    st.pyplot(bbox_inches="tight", clear_figure=True)
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")
        else:
            st.info("Install `shap` to see SHAP-based local and global explanations.")

        # What‑if analysis: allow editing numeric/categorical and re‑predict
        st.markdown("**What‑if analysis (counterfactual sliderboard)**")
        edited = instance_raw.copy()
        with st.form("what_if"):
            cols1, cols2 = st.columns(2)
            with cols1:
                for col in [c for c in instance_raw.columns if c in numeric_cols]:
                    val = float(instance_raw.iloc[0][col])
                    mn = float(df[col].quantile(0.01)) if np.isfinite(df[col].quantile(0.01)) else val - 5
                    mx = float(df[col].quantile(0.99)) if np.isfinite(df[col].quantile(0.99)) else val + 5
                    edited[col] = st.slider(f"{col}", min_value=mn, max_value=mx, value=val)
            with cols2:
                for col in [c for c in instance_raw.columns if c in categorical_cols]:
                    choices = sorted(df[col].dropna().astype(str).unique().tolist())
                    current = str(instance_raw.iloc[0][col])
                    edited[col] = st.selectbox(f"{col}", choices=choices, index=choices.index(current) if current in choices else 0)
            submitted = st.form_submit_button("Recompute prediction")
        if submitted:
            Xe = preprocessor.transform(edited)
            new_pred = baseline.predict(Xe)[0]
            new_label = inverse_target_mapping[new_pred]
            if hasattr(baseline, "predict_proba"):
                new_proba = float(baseline.predict_proba(Xe)[0][new_pred])
                st.success(f"**New prediction:** {new_label}  •  P={new_proba:.3f}")
            else:
                st.success(f"**New prediction:** {new_label}")
            st.caption("Compare with original to see which edits helped/hurt the favorable outcome.")
else:
    st.info("Test set is empty; cannot show local explanations.")

# -----------------------------
# 🩹 Bias mitigation via simple reweighing
# -----------------------------
st.subheader("🩹 Bias Mitigation (Reweighing + RandomForest)")
st.caption("Weights groups so the intersectional distribution is more uniform within each class label. Uses groups among available protected attributes.")

available_for_weighting = [c for c in ['Gender','Marital status','AgeBin','Scholarship holder','Displaced','Debtor'] if c in X_train_orig.columns]
if not available_for_weighting:
    st.info("No protected attributes available for reweighing. Skipping mitigation.")
else:
    train_df = X_train_orig.copy()
    train_df['Target'] = y_train.copy()

    group_cols = available_for_weighting
    grouped = train_df.groupby(group_cols + ['Target']).size().rename('n').reset_index()
    weights = pd.Series(1.0, index=train_df.index, dtype='float64')

    for tval, gdf in grouped.groupby('Target'):
        total_t = (train_df['Target'] == tval).sum()
        if total_t == 0:
            continue
        k = len(gdf)
        target_prob = 1.0 / max(k, 1)
        for _, row in gdf.iterrows():
            mask = (train_df['Target'] == tval)
            for c in group_cols:
                mask &= (train_df[c] == row[c])
            count = mask.sum()
            if count == 0:
                continue
            p = count / total_t
            w = target_prob / max(p, 1e-6)
            weights.loc[mask] = w
    weights *= len(weights) / weights.sum()

    with st.status("Training mitigated RandomForest...", expanded=False) as status:
        mitigated = RandomForestClassifier(random_state=rand_state)
        mitigated.fit(X_train, y_train, sample_weight=weights.values)
        y_pred_m = mitigated.predict(X_test)
        status.update(label="Mitigated model trained", state="complete")

    acc_m = accuracy_score(y_test, y_pred_m)
    st.metric("Mitigated Accuracy", f"{acc_m:.4f}")
    cr_m = classification_report(y_test, y_pred_m, output_dict=True, zero_division=0)
    with st.expander("Classification report (mitigated)"):
        st.json(cr_m)

    y_pred_m_cat = pd.Series(y_pred_m, index=X_test_orig.index).map(inverse_target_mapping)

    st.subheader("⚖️ Fairness (Mitigated)")
    for attr in prot_for_eval:
        st.markdown(f"**Attribute:** `{attr}`")
        groups = X_test_orig[attr]
        rates, di, dpd = disparate_impact_and_dpd(y_pred_m_cat, groups, favorable_label=favorable_label)
        if rates is not None:
            df_rates = pd.DataFrame({"pred_favorable_rate": rates}).sort_index()
            if di is not None:
                df_rates["disparate_impact_vs_best"] = di
                df_rates["dpd_vs_best"] = dpd
            st.dataframe(df_rates.style.format("{:.3f}"), use_container_width=True)
        tprs, eo = equal_opportunity(y_test_cat, y_pred_m_cat, groups, target_mapping, favorable_label=favorable_label)
        if tprs is not None:
            df_eo = pd.DataFrame({"TPR": tprs, "EO_diff_vs_best": eo}).sort_index()
            st.dataframe(df_eo.style.format("{:.3f}"), use_container_width=True)

# -----------------------------
# 📤 Export
# -----------------------------
st.subheader("📤 Export")
col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    if st.button("Export baseline classification report JSON"):
        st.download_button("Download JSON", data=json.dumps(cr_baseline, indent=2), file_name="classification_report_baseline.json", mime="application/json")
with col_exp2:
    if importances is not None and len(feat_names_full) == len(importances):
        csv_bytes = aggregate_importances_to_base(importances, feat_names_full).to_csv().encode()
        st.download_button("Download global importances (CSV)", data=csv_bytes, file_name="global_importances.csv", mime="text/csv")

st.info("Tip: Use the sidebar to add your GitHub repo and hosted app links. Version label: " + APP_VERSION)
