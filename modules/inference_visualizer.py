# file: modules/inference_visualizer.py
import os
import shutil
import yaml
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
from modules.fuzzy_area_classifier import FuzzyAreaClassifier # Impor kelas baru

class InferenceVisualizer:
    """
    Mengelola inferensi pada gambar uji dan visualisasi hasilnya untuk Instance Segmentation.
    """
    def __init__(self, model_dir, img_size):
        self.MODEL_DIR = model_dir
        self.img_size = img_size
        self.INFERENCE_OUTPUT_DIR = os.path.join(self.MODEL_DIR, "inference_outputs")
        os.makedirs(self.INFERENCE_OUTPUT_DIR, exist_ok=True)
        # Menggunakan kelas FuzzyAreaClassifier sebagai dependensi
        self.fuzzy_classifier = FuzzyAreaClassifier()

    def run_inference_on_sample_images(self, model_path, data_yaml_path="", num_images=4, conf_threshold=0.25):
        """
        Menjalankan inferensi pada beberapa gambar uji dan mengumpulkan hasilnya untuk segmentasi.
        """
        print(f"\n--- Menjalankan inferensi pada gambar uji dengan model {model_path} (segment) ---")
        if not os.path.exists(model_path):
            print(f"Error: Model tidak ditemukan di {model_path}. Tidak dapat menjalankan inferensi.")
            return None

        model = YOLO(model_path)
        
        test_images_dir = ""
        try:
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            dataset_root = data.get('path')
            test_path_in_yaml = data.get('test')
            
            if os.path.isdir(os.path.join(dataset_root, test_path_in_yaml, 'images')):
                test_images_dir = os.path.join(dataset_root, test_path_in_yaml, 'images')
            elif os.path.isdir(os.path.join(dataset_root, test_path_in_yaml)):
                test_images_dir = os.path.join(dataset_root, test_path_in_yaml)
            
            class_names = data.get('names')
        except Exception as e:
            print(f"Error membaca data.yaml atau path gambar uji: {e}")
            return None

        if not os.path.exists(test_images_dir):
            print(f"Error: Direktori gambar uji tidak ditemukan di {test_images_dir}.")
            return None

        image_files = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("Tidak ada gambar uji ditemukan di direktori.")
            return None

        np.random.shuffle(image_files)
        sample_images = image_files[:num_images]

        inference_results = []
        for img_path in sample_images:
            print(f"Inferensi pada: {os.path.basename(img_path)}")
            
            original_img = cv2.imread(img_path)
            if original_img is None:
                print(f"Peringatan: Gagal membaca gambar {img_path}. Melewatkan.")
                continue
            original_h, original_w, _ = original_img.shape
            
            results = model.predict(img_path, conf=conf_threshold, imgsz=self.img_size[0], verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else np.array([])
                scores = r.boxes.conf.cpu().numpy() if r.boxes else np.array([])
                class_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else np.array([])
                
                masks = None
                total_mask_area = 0
                normalized_mask_area = 0.0
                fuzzy_area_classification = "N/A"

                if r.masks:
                    masks = r.masks.data.cpu().numpy()
                    for mask in masks:
                        mask_resized_to_original = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        total_mask_area += np.sum(mask_resized_to_original > 0)
                    
                    if (original_w * original_h) > 0:
                        normalized_mask_area = (total_mask_area / (original_w * original_h)) * 100
                        # Memanggil kelas FuzzyAreaClassifier yang terpisah
                        fuzzy_area_classification = self.fuzzy_classifier.classify_area(normalized_mask_area)

                num_objects = len(boxes)

                inference_results.append({
                    'image_path': img_path,
                    'boxes': boxes,
                    'scores': scores,
                    'class_ids': class_ids,
                    'masks': masks,
                    'num_objects': num_objects,
                    'total_mask_area_px': total_mask_area,
                    'normalized_mask_area_percent': normalized_mask_area,
                    'fuzzy_area_classification': fuzzy_area_classification,
                    'class_names': class_names,
                    'original_width': original_w,
                    'original_height': original_h
                })
        return inference_results

    def visualize_inference_results_grid(self, inference_results, title="Inference Results"):
        """
        Memvisualisasikan hasil inferensi segmentasi dalam grid Matplotlib.
        """
        if not inference_results:
            print("Tidak ada hasil inferensi untuk divisualisasikan.")
            return

        num_images = len(inference_results)
        cols = min(4, num_images)
        rows = math.ceil(num_images / cols)

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), dpi=150)
        axes = axes.flatten()

        fig.suptitle(title, fontsize=18)

        for i, result in enumerate(inference_results):
            if i >= len(axes):
                break

            ax = axes[i]
            img = cv2.imread(result['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img)
            ax.axis('off')

            combined_mask = np.zeros_like(img, dtype=np.uint8)
            alpha = 0.4
            
            for j, mask_data in enumerate(result['masks']):
                mask_resized = cv2.resize(mask_data.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                color = [np.random.randint(0, 255) for _ in range(3)]
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask_resized > 0] = color
                combined_mask = cv2.addWeighted(combined_mask, 1, colored_mask, alpha, 0)
            
            final_img = cv2.addWeighted(img, 1, combined_mask, 1, 0)
            ax.imshow(final_img)

            for j, box in enumerate(result['boxes']):
                x1, y1, x2, y2 = map(int, box)
                class_id = result['class_ids'][j]
                score = result['scores'][j]
                label = f"{result['class_names'][class_id]}: {score:.2f}"
                
                bbox_color = (255, 0, 0)
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            fill=False, edgecolor=bbox_color, linewidth=2))
                ax.text(x1, y1 - 10, label, color='white', fontsize=8,
                        bbox=dict(facecolor=bbox_color, edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

            info_text = (f"Objects: {result['num_objects']}\n"
                         f"Area: {result['total_mask_area_px']:.0f} px ({result['normalized_mask_area_percent']:.2f}%)\n"
                         f"Fuzzy: {result['fuzzy_area_classification']}")
            ax.set_title(f"Image {i+1}\n{info_text}", fontsize=10)

        for j in range(num_images, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        plot_filename = f"inference_results_{title.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(self.INFERENCE_OUTPUT_DIR, plot_filename), dpi=150)
        print(f"Plot hasil inferensi disimpan ke: {os.path.join(self.INFERENCE_OUTPUT_DIR, plot_filename)}")
        plt.show()

    def save_superimposed_images(self, inference_results, model_version):
        """
        Menyimpan gambar hasil inferensi segmentasi yang telah disuperimpose.
        """
        if not inference_results:
            print("Tidak ada hasil inferensi untuk disimpan.")
            return

        output_folder = os.path.join(self.INFERENCE_OUTPUT_DIR, f"yolo{model_version}_segment_results")
        os.makedirs(output_folder, exist_ok=True)
        print(f"\n--- Menyimpan gambar hasil inferensi ke: {output_folder} ---")

        for i, result in enumerate(inference_results):
            img = cv2.imread(result['image_path'])
            combined_mask_overlay = np.zeros_like(img, dtype=np.uint8)
            alpha = 0.4
            
            for j, mask_data in enumerate(result['masks']):
                mask_resized = cv2.resize(mask_data.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                color = [np.random.randint(0, 255) for _ in range(3)]
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask_resized > 0] = color
                combined_mask_overlay = cv2.addWeighted(combined_mask_overlay, 1, colored_mask, alpha, 0)
            
            final_img = cv2.addWeighted(img, 1, combined_mask_overlay, 1, 0)

            for j, box in enumerate(result['boxes']):
                x1, y1, x2, y2 = map(int, box)
                class_id = result['class_ids'][j]
                score = result['scores'][j]
                label = f"{result['class_names'][class_id]}: {score:.2f}"
                
                bbox_color = (0, 255, 0)
                cv2.rectangle(final_img, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.putText(final_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2, cv2.LINE_AA)
            
            output_filepath = os.path.join(output_folder, os.path.basename(result['image_path']))
            cv2.imwrite(output_filepath, final_img)
            print(f"  Saved: {os.path.basename(output_filepath)}")

    def zip_superimposed_images_folder(self, model_version):
        """
        Mengkompres folder yang berisi gambar hasil superimpose menjadi file zip.
        """
        source_folder = os.path.join(self.INFERENCE_OUTPUT_DIR, f"yolo{model_version}_segment_results")
        output_filename = f"yolo{model_version}_segment_inference_results"
        if os.path.exists(source_folder):
            print(f"\n--- Mengkompresi folder hasil inferensi untuk YOLO{model_version} (segmentasi) ---")
            try:
                shutil.make_archive(os.path.join(self.INFERENCE_OUTPUT_DIR, output_filename), 'zip', source_folder)
                print(f"Folder '{source_folder}' berhasil dikompresi menjadi '{output_filename}.zip'")
            except Exception as e:
                print(f"Gagal mengkompresi folder hasil inferensi '{source_folder}': {e}")
        else:
            print(f"Peringatan: Folder hasil inferensi '{source_folder}' tidak ditemukan. Tidak dapat mengkompresi.")

    def save_inference_results_csv(self, inference_results, model_version):
        """
        Menyimpan ringkasan hasil inferensi ke file CSV.
        """
        if not inference_results:
            print("Tidak ada hasil inferensi untuk disimpan ke CSV.")
            return

        output_folder = os.path.join(self.INFERENCE_OUTPUT_DIR, f"yolo{model_version}_segment_results")
        os.makedirs(output_folder, exist_ok=True)

        csv_data = []
        for result in inference_results:
            row = {
                "Image Name": os.path.basename(result['image_path']),
                "Num Objects Detected": result['num_objects'],
                "Total Mask Area (px)": f"{result['total_mask_area_px']:.0f}",
                "Normalized Mask Area (%)": f"{result['normalized_mask_area_percent']:.2f}",
                "Fuzzy Area Classification": result['fuzzy_area_classification']
            }
            csv_data.append(row)
        
        df_results = pd.DataFrame(csv_data)
        csv_filename = f"inference_summary_yolo{model_version}_segment.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)
        df_results.to_csv(csv_filepath, index=False)
        print(f"Ringkasan hasil inferensi disimpan ke: {csv_filepath}")