# file: modules/rknn_converter.py
import os
import shutil

try:
    from rknn.api import RKNN
except ImportError:
    print("RKNN Toolkit2 tidak ditemukan. Konversi RKNN mungkin tidak dapat dilakukan di lingkungan ini.")
    RKNN = None

class RknnConverter:
    """
    Mengelola konversi model ONNX ke format RKNN.
    """
    def __init__(self, rknn_model_dir, img_size):
        self.RKNN_MODEL_DIR = rknn_model_dir
        self.img_size = img_size
        os.makedirs(self.RKNN_MODEL_DIR, exist_ok=True)
        
        self.YOLOV8N_IS_RKNN_PATH = os.path.join(self.RKNN_MODEL_DIR, "yolov8n_is.rknn")
        self.YOLOV10N_IS_RKNN_PATH = os.path.join(self.RKNN_MODEL_DIR, "yolov10n_is.rknn")
        self.YOLOV11N_IS_RKNN_PATH = os.path.join(self.RKNN_MODEL_DIR, "yolov11n_is.rknn")

    def convert_onnx_to_rknn(self, onnx_model_path, model_version):
        """
        Mengkonversi model ONNX ke format RKNN.
        """
        if RKNN is None:
            print("RKNN Toolkit2 tidak terinstal. Lewati konversi model.")
            return False

        rknn_model_output_path = ""
        if model_version == "v8n":
            rknn_model_output_path = self.YOLOV8N_IS_RKNN_PATH
        elif model_version == "v10n":
            rknn_model_output_path = self.YOLOV10N_IS_RKNN_PATH
        elif model_version == "v11n":
            rknn_model_output_path = self.YOLOV11N_IS_RKNN_PATH
        else:
            print(f"Error: Versi model {model_version} tidak dikenal untuk konversi RKNN.")
            return False
            
        print(f"\n--- Mengkonversi Model ONNX ({model_version}) ke RKNN ---")
        if not os.path.exists(onnx_model_path):
            print(f"Error: Model ONNX tidak ditemukan di {onnx_model_path}.")
            return False

        rknn = RKNN(verbose=True)
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform='rk3588',
        )

        ret = rknn.load_onnx(model=onnx_model_path)
        if ret != 0:
            print('Gagal memuat model ONNX!')
            return False

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print('Gagal membangun model RKNN!')
            return False

        ret = rknn.export_rknn(rknn_model_output_path)
        if ret != 0:
            print('Gagal mengekspor model RKNN!')
            return False

        print("Konversi model ke RKNN berhasil!")
        rknn.release()
        return True

    def zip_rknn_models_folder(self):
        """
        Melakukan kompresi (zip) pada folder 'rknn_models'.
        """
        source_folder = self.RKNN_MODEL_DIR
        output_filename = "rknn_models"
        if os.path.exists(source_folder):
            print(f"\n--- Mengkompresi folder RKNN models ---")
            try:
                shutil.make_archive(output_filename, 'zip', source_folder)
                print(f"Folder '{source_folder}' berhasil dikompresi menjadi '{output_filename}.zip'")
            except Exception as e:
                print(f"Gagal mengkompresi folder RKNN models '{source_folder}': {e}")
        else:
            print(f"Peringatan: Folder RKNN models '{source_folder}' tidak ditemukan. Tidak dapat mengkompresi.")
