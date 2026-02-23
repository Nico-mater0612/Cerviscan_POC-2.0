import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import pickle
import numpy as np
from Orange.data import Table

# -----------------------------
# UI setup
# -----------------------------
st.set_page_config(page_title="CerviScan AI", page_icon="🔬", layout="centered")
st.title("🔬 P & C's CerviScan Diagnostic Engine (PoC)")

st.info(
    "Research proof-of-concept only (not a medical diagnosis). "
    "Model expects **RMA log2 mRNA expression** values (typical range ~2–15). "
    "Different lab platforms/normalization can shift values and change the score."
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    with open("Cervicalcancer_randomforest_P&C.pkcls", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("Model failed to load. Make sure the .pkcls file is in the same folder and sklearn==1.5.2 is installed.")
    st.exception(e)
    st.stop()

# -----------------------------
# Genes (UI)
# -----------------------------
genes = ['CCNB2', 'CDC20', 'CDKN2A', 'CDKN3', 'MCM2', 'MKI67', 'NUSAP1', 'PRC1', 'TOP2A', 'VEGFA']

# -----------------------------
# Session state for inputs
# -----------------------------
if "inputs" not in st.session_state:
    st.session_state.inputs = {g: 7.0 for g in genes}

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Patient Info (optional)")
patient_id = st.sidebar.text_input("Patient ID (optional)", value="")

st.sidebar.header("Interpretation Bands (PoC)")
low_cut = st.sidebar.slider("Low / Borderline cut", 0.0, 1.0, 0.30, 0.01)
high_cut = st.sidebar.slider("Borderline / High cut", 0.0, 1.0, 0.70, 0.01)

show_debug = st.sidebar.checkbox("Show debug", value=False)

# -----------------------------
# Prediction helpers (handles required meta column)
# -----------------------------
def _make_table_from_inputs(inputs_dict):
    domain = model.domain
    expected = [a.name for a in domain.attributes]

    missing = [name for name in expected if name not in inputs_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = np.array([[float(inputs_dict[name]) for name in expected]], dtype=np.float64)
    if np.isnan(X).any():
        raise ValueError("Input contains NaN values.")

    Y = np.array([np.nan], dtype=np.float64)  # unknown class

    # REQUIRED metas (your model domain expects 1 meta col, e.g., patient ID)
    M = np.empty((1, len(domain.metas)), dtype=object)
    M[:] = ""
    if len(domain.metas) >= 1:
        M[0, 0] = patient_id  # can be blank

    data = Table.from_numpy(domain, X=X, Y=Y, metas=M)
    return data, expected, X, M

def predict_probs(inputs_dict):
    data, expected, X, M = _make_table_from_inputs(inputs_dict)
    probs = model(data, model.Probs)[0]
    classes = list(model.domain.class_var.values)
    return probs, classes, expected, X, M

def abnormal_probability(probs, classes):
    # Your class order was: ['Abnormal', 'Normal']
    if "Abnormal" in classes:
        idx = classes.index("Abnormal")
    else:
        idx = 0
    return float(probs[idx])

# -----------------------------
# DEMO MODE (Guaranteed buttons)
# Generates profiles that the model itself scores ~10% and ~90% abnormal.
# -----------------------------
st.sidebar.header("Demo Mode (Guaranteed)")

def find_scalar_for_target(target_prob, lo=2.0, hi=15.0, steps=30):
    """
    Find scalar s in [lo, hi] such that setting all genes to s gives
    abnormal_prob ~ target_prob. Assumes monotonic-ish behavior.
    """
    # endpoint probs
    inp_lo = {g: lo for g in genes}
    inp_hi = {g: hi for g in genes}

    probs_lo, classes_lo, _, _, _ = predict_probs(inp_lo)
    probs_hi, classes_hi, _, _, _ = predict_probs(inp_hi)

    p_lo = abnormal_probability(probs_lo, classes_lo)
    p_hi = abnormal_probability(probs_hi, classes_hi)

    increasing = (p_hi >= p_lo)

    left, right = lo, hi
    for _ in range(steps):
        mid = (left + right) / 2.0
        inp_mid = {g: mid for g in genes}
        probs_mid, classes_mid, _, _, _ = predict_probs(inp_mid)
        p_mid = abnormal_probability(probs_mid, classes_mid)

        if increasing:
            if p_mid < target_prob:
                left = mid
            else:
                right = mid
        else:
            if p_mid > target_prob:
                left = mid
            else:
                right = mid

    return (left + right) / 2.0

colA, colB = st.sidebar.columns(2)

if colA.button("Load Normal"):
    s = find_scalar_for_target(0.10)  # ~10% abnormal
    for g in genes:
        st.session_state.inputs[g] = float(s)

if colB.button("Load Abnormal"):
    s = find_scalar_for_target(0.90)  # ~90% abnormal
    for g in genes:
        st.session_state.inputs[g] = float(s)

st.sidebar.caption("These presets are model-calibrated demo points (not real patients).")

# -----------------------------
# Sidebar gene inputs
# -----------------------------
st.sidebar.header("Gene Expression (RMA log2)")
st.sidebar.caption("Restricted to 2.0–15.0 to match the training space.")

for gene in genes:
    st.session_state.inputs[gene] = st.sidebar.number_input(
        gene,
        min_value=2.0,
        max_value=15.0,
        value=float(st.session_state.inputs[gene]),
        step=0.1
    )

# -----------------------------
# Run prediction
# -----------------------------
if st.button("RUN DIAGNOSTIC"):
    try:
        probs, classes, expected, X, M = predict_probs(st.session_state.inputs)
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()

    ab_prob = abnormal_probability(probs, classes)
    ab_pct = ab_prob * 100.0
    normal_pct = (1.0 - ab_prob) * 100.0 if len(classes) == 2 else max(0.0, 100.0 - ab_pct)

    st.divider()

    # 3-tier PoC interpretation
    if ab_prob < low_cut:
        st.success("### ✅ Low Risk (PoC)")
        st.write(f"AI Risk Score (Abnormal probability): **{ab_pct:.1f}%**")
        st.caption("Closer to normal patterns in the training set.")
    elif ab_prob < high_cut:
        st.warning("### ⚠️ Borderline / Needs Review (PoC)")
        st.write(f"AI Risk Score (Abnormal probability): **{ab_pct:.1f}%**")
        st.caption("Intermediate score. Could reflect variation, inflammation/HPV response, or measurement differences.")
    else:
        st.error("### 🚨 High Risk (PoC)")
        st.write(f"AI Risk Score (Abnormal probability): **{ab_pct:.1f}%**")
        st.caption("Strong match to abnormal patterns in the training set.")

    st.progress(min(max(ab_prob, 0.0), 1.0))

    c1, c2 = st.columns(2)
    c1.metric("Abnormal (Risk)", f"{ab_pct:.1f}%")
    c2.metric("Normal (Healthy)", f"{normal_pct:.1f}%")

    if show_debug:
        with st.expander("Debug"):
            st.write("Model type:", type(model))
            st.write("Class order:", classes)
            st.write("Expected feature order:", expected)
            st.write("Metas expected:", [m.name for m in model.domain.metas])
            st.write("X sent:", X)
            st.write("Metas sent:", M)
            st.write("Raw probs:", probs.tolist() if hasattr(probs, "tolist") else probs)
