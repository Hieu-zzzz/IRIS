from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import random
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


@st.cache_resource(show_spinner=False)
def train_cached_model(seed: int) -> Tuple[MLPClassifier, StandardScaler, np.ndarray]:
    iris = load_iris()
    features = iris.data
    targets = iris.target

    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=0.2, random_state=seed, stratify=targets
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    parameter_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (64, 32), (128, 64)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-2, 1e-3, 5e-4],
    }

    base_mlp = MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=False,
        random_state=seed,
    )

    search = GridSearchCV(
        estimator=base_mlp,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    search.fit(features_train_scaled, targets_train)
    model = search.best_estimator_

    predictions = model.predict(features_test_scaled)
    accuracy = accuracy_score(targets_test, predictions)

    st.session_state["iris_target_names"] = load_iris().target_names
    st.session_state["classification_report_text"] = classification_report(
        targets_test, predictions, target_names=load_iris().target_names
    )
    st.session_state["best_params"] = search.best_params_

    return model, scaler, features_test_scaled


def render_app() -> None:
    st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

    custom_css = """
    <style>
    :root { --primary:#ec4899; --primary-dark:#db2777; --accent:#fce7f3; }
    .title {font-size: 2.2rem; font-weight: 700; margin-bottom: .25rem; color: var(--primary)}
    .subtitle {color:#6b7280; margin-bottom: 1rem}
    .card {background: #ffffff; border-radius: 14px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,.06); border: 1px solid #f9a8d4}
    .metric {font-size: 1.7rem; font-weight: 700; color: var(--primary-dark)}
    .stButton>button {background: linear-gradient(90deg, var(--primary), var(--primary-dark)); color: #fff; border: 0; border-radius: 10px}
    .stButton>button:hover {filter: brightness(0.95)}
    .stSuccess {background: var(--accent) !important; color: #9d174d !important}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="title">IRIS </div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Dự đoán loài hoa từ 4 thông số</div>', unsafe_allow_html=True)

    # Controls for retraining so accuracy can change (not fixed)
    with st.sidebar:
        st.header("Thiết lập mô hình")
        default_seed = st.session_state.get("seed", 42)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=int(default_seed), step=1)
        retrain = st.button("Huấn luyện lại", use_container_width=True)
        if retrain:
            st.session_state["seed"] = int(seed)
            st.cache_resource.clear()
            st.rerun()

    current_seed = int(st.session_state.get("seed", 42))
    with st.spinner("Đang huấn luyện mô hình..."):
        model, scaler, _ = train_cached_model(current_seed)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Nhập thông số")
        sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

        if st.button("Dự đoán", use_container_width=True):
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_features)
            pred_class = model.predict(input_scaled)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]
            target_names = st.session_state.get("iris_target_names")
            st.success(f"Loài dự đoán: {target_names[pred_class].capitalize()}")
            # Map predicted species to a local image and save to session for the right panel
            species = target_names[pred_class]
            species_to_image = {
                "setosa": "irisseto2.jpg",
                "versicolor": "acuaticas-iris-versicolor-12_2.jpg",
                "virginica": "0001268_southern-blue-flag-iris-iris-virginica.jpeg",
            }
            image_path = species_to_image.get(species)
            if image_path and os.path.exists(image_path):
                st.session_state["predicted_image_path"] = image_path
                st.session_state["predicted_species_name"] = species
            else:
                st.session_state.pop("predicted_image_path", None)
                st.session_state.pop("predicted_species_name", None)
            if 'proba' in locals():
                st.caption("Độ tự tin dự đoán")
                # Show numeric table and chart
                df = pd.DataFrame({
                    "Class": list(target_names),
                    "Probability": proba
                })
                col_tbl, col_chart = st.columns([1,1])
                with col_tbl:
                    st.dataframe(df.style.format({"Probability": "{:.3f}"}), use_container_width=True)
                with col_chart:
                    st.bar_chart(df, x="Class", y="Probability")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Tải ảnh minh hoạ (tùy chọn)")
        uploaded = st.file_uploader("Chọn ảnh bông hoa (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Ảnh bạn chọn", use_column_width=True)
        else:
            # If user didn't upload, show the image matching the predicted class (if available)
            pred_img_path = st.session_state.get("predicted_image_path")
            pred_species = st.session_state.get("predicted_species_name")
            if pred_img_path and os.path.exists(pred_img_path):
                st.image(pred_img_path, caption=f"Ảnh minh hoạ: {pred_species.capitalize()}", use_column_width=True)
            else:
                st.info("Bạn có thể tải ảnh để hiển thị minh hoạ hoặc bấm Dự đoán để xem ảnh tương ứng.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    metrics = st.columns(3)
    with metrics[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Best Params")
        st.code(str(st.session_state.get("best_params")), language="txt")
        st.markdown('</div>', unsafe_allow_html=True)
    with metrics[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Độ chính xác (test)")
        # Recompute accuracy with cached artifacts for display
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
        )
        scaler2 = StandardScaler().fit(X_train)
        preds = model.predict(scaler2.transform(X_test))
        acc = accuracy_score(y_test, preds)
        st.markdown(f'<div class="metric">{acc:.3f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with metrics[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Báo cáo phân loại")
        st.code(st.session_state.get("classification_report_text", ""), language="txt")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    render_app()

