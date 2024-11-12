import tensorflow as tf
import tf2onnx
import os
import kagglehub
from tensorflow.keras.models import load_model, Model
import onnx
import traceback


def download_model(model_identifier, save_path):
    """Download a model file if it doesn't exist."""
    if not os.path.exists(save_path):
        try:
            print(f"Downloading model to {save_path}...")
            model_path = kagglehub.model_download(model_identifier)

            # List files in the downloaded directory
            downloaded_files = os.listdir(model_path)
            if not downloaded_files:
                raise FileNotFoundError(f"No files found in {model_path}")

            # Take the first file if there's only one, or look for .h5 file
            source_file = None
            for file in downloaded_files:
                if file.endswith(".h5"):
                    source_file = file
                    break
            if not source_file:
                source_file = downloaded_files[0]

            # Copy the file
            source_path = os.path.join(model_path, source_file)
            print(f"Copying {source_path} to {save_path}")
            import shutil

            shutil.copy2(source_path, save_path)
        except kagglehub.exceptions.KaggleApiHTTPError as e:
            print(
                f"Authentication Error: Please ensure you have set up your Kaggle API credentials."
            )
            print("1. Go to https://www.kaggle.com/settings")
            print("2. Scroll to API section and click 'Create New API Token'")
            print("3. Place the downloaded kaggle.json file in ~/.kaggle/ directory")
            raise
    return save_path


def get_model_input_shape(model):
    """Safely extract input shape from model."""
    try:
        # Try to get input shape from model config
        if hasattr(model, "get_config"):
            config = model.get_config()
            if "layers" in config and len(config["layers"]) > 0:
                first_layer = config["layers"][0]
                if (
                    "config" in first_layer
                    and "batch_input_shape" in first_layer["config"]
                ):
                    return first_layer["config"]["batch_input_shape"]

        # Fallback: try to build the model to get input shape
        if not model.built:
            model.build()
        if hasattr(model, "input_shape"):
            return model.input_shape

        # Last resort: use default shape for image classification
        return (None, 224, 224, 3)
    except Exception as e:
        print(f"Warning: Could not determine input shape: {e}")
        return (None, 224, 224, 3)


def convert_to_functional_model(sequential_model):
    """Convert a Sequential model to a Functional model."""
    try:
        # Get input shape
        input_shape = get_model_input_shape(sequential_model)
        print(f"Using input shape: {input_shape}")

        # Create input layer
        input_layer = tf.keras.layers.Input(shape=input_shape[1:])

        # Build the functional model
        x = input_layer
        for layer in sequential_model.layers:
            x = layer(x)

        functional_model = Model(inputs=[input_layer], outputs=[x])
        functional_model.set_weights(sequential_model.get_weights())
        return functional_model
    except Exception as e:
        print(f"Error in converting to functional model: {e}")
        traceback.print_exc()
        raise


def convert_tf_to_onnx(model, output_path):
    """Convert TensorFlow model to ONNX format."""
    try:
        # Convert Sequential to Functional if necessary
        if isinstance(model, tf.keras.Sequential):
            print("Converting Sequential model to Functional model...")
            model = convert_to_functional_model(model)

        # Ensure model is compiled
        if not model.compiled_loss:
            print("Compiling model...")
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

        # Get input shape from model
        input_shape = get_model_input_shape(model)
        print(f"Creating input signature with shape: {input_shape}")

        # Create input signature
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

        # Convert to ONNX
        print("Converting to ONNX...")
        output_path = output_path.replace("\\", "/")  # Normalize path for Windows
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=13, output_path=output_path
        )

        return True
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        traceback.print_exc()
        return False


def download_and_convert_models():
    """Download and convert models to ONNX format."""
    try:
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        model_configs = {
            "cnn": {
                "identifier": "mikeodnis/brain_tumor_cnn/tensorFlow2/default",
                "output": os.path.join(models_dir, "cnn_model.onnx"),
                "keras_path": os.path.join(models_dir, "cnn_model.h5"),
            },
            "xception": {
                "identifier": "mikeodnis/brain_tumor_cnn/tensorFlow2/default",
                "output": os.path.join(models_dir, "xception_model.onnx"),
                "keras_path": os.path.join(models_dir, "xception_model.h5"),
            },
        }

        for model_name, config in model_configs.items():
            print(f"\nProcessing {model_name} model...")

            # Download the model first
            keras_path = download_model(config["identifier"], config["keras_path"])

            # Load the model directly using load_model
            print(f"Loading {model_name} model...")
            model = load_model(keras_path)

            # Convert to ONNX
            print(f"Converting {model_name} model to ONNX format...")
            success = convert_tf_to_onnx(model, config["output"])

            if success:
                print(
                    f"{model_name} model successfully converted and saved to {config['output']}"
                )

                # Verify the ONNX model
                try:
                    onnx_model = onnx.load(config["output"])
                    onnx.checker.check_model(onnx_model)
                    print(f"{model_name} ONNX model verification successful!")
                except Exception as e:
                    print(f"{model_name} ONNX model verification failed: {str(e)}")

    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    download_and_convert_models()
