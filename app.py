import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Satellite Image Classification Benchmark", layout="wide")

st.title("🛰️ Deep Learning for Satellite Image Classification")
st.markdown("### Benchmarking CNN Models on Satellite Dataset")

DATASET_DIR = "satellite_dataset"

batch_size = st.sidebar.slider("Batch Size", 8, 64, 16)
epochs = st.sidebar.slider("Epochs", 1, 20, 5)

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(64,64),
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=(64,64),
    batch_size=batch_size,
    class_mode='categorical'
)

model = build_model(train_gen.num_classes)

if st.button("🚀 Train Model"):
    with st.spinner("Training in progress..."):
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=test_gen
        )

    st.success("Training Completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy")
        plt.plot(history.history['accuracy'], label="Train")
        plt.plot(history.history['val_accuracy'], label="Validation")
        plt.legend()
        st.pyplot(plt)

    with col2:
        st.subheader("Loss")
        plt.plot(history.history['loss'], label="Train")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.legend()
        st.pyplot(plt)

    loss, acc = model.evaluate(test_gen)
    st.metric("Test Accuracy", f"{acc*100:.2f}%")
