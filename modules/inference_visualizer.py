# file: modules/inference_visualizer.py
import os
import shutil
import yaml
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
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

    def visualize_inference_results_grid(self, inference_results, title="Inference Results", save_only=False):
        """
        Memvisualisasikan hasil inferensi segmentasi dalam grid Matplotlib.
        Fixed RGBA color issues and improved for academic paper quality.
        
        Args:
            inference_results: List of inference results
            title: Title for the visualization
            save_only: If True, only save to file without showing (useful for headless environments)
        """
        if not inference_results:
            print("Tidak ada hasil inferensi untuk divisualisasikan.")
            return None

        num_images = len(inference_results)
        cols = min(4, num_images)
        rows = math.ceil(num_images / cols)

        try:
            # Set matplotlib backend to Agg for better compatibility
            import matplotlib
            if save_only:
                matplotlib.use('Agg')
            
            # Use a clean, academic style
            plt.style.use('default')  # More reliable than seaborn
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), dpi=150, 
                                   facecolor='white', edgecolor='black')
            
            # Handle single subplot case
            if num_images == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            fig.suptitle(title, fontsize=18, fontweight='bold')

            # Define a set of distinct colors in 0-1 range for matplotlib
            colors_01 = [
                [1.0, 0.0, 0.0],    # Red
                [0.0, 1.0, 0.0],    # Green  
                [0.0, 0.0, 1.0],    # Blue
                [1.0, 1.0, 0.0],    # Yellow
                [1.0, 0.0, 1.0],    # Magenta
                [0.0, 1.0, 1.0],    # Cyan
                [1.0, 0.5, 0.0],    # Orange
                [0.5, 0.0, 1.0],    # Purple
                [0.0, 0.5, 0.0],    # Dark Green
                [0.5, 0.5, 0.5]     # Gray
            ]

            for i, result in enumerate(inference_results):
                if i >= len(axes):
                    break

                ax = axes[i]
                
                # Load and prepare image
                img = cv2.imread(result['image_path'])
                if img is None:
                    print(f"Warning: Could not load image {result['image_path']}")
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Normalize image to 0-1 range for matplotlib
                img_normalized = img_rgb.astype(np.float32) / 255.0
                
                # Create overlay for masks
                overlay = np.zeros_like(img_normalized)
                
                # Process masks if they exist
                if result['masks'] is not None and len(result['masks']) > 0:
                    for j, mask_data in enumerate(result['masks']):
                        # Resize mask to match image dimensions
                        mask_resized = cv2.resize(mask_data.astype(np.uint8), 
                                                (img_rgb.shape[1], img_rgb.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                        
                        # Use predefined colors in 0-1 range
                        color = colors_01[j % len(colors_01)]
                        
                        # Create colored mask
                        mask_bool = mask_resized > 0
                        overlay[mask_bool] = color
                
                # Blend original image with mask overlay
                alpha = 0.4
                blended = img_normalized * (1 - alpha) + overlay * alpha
                
                # Ensure values are in valid range
                blended = np.clip(blended, 0, 1)
                
                # Display the blended image
                ax.imshow(blended)
                ax.axis('off')

                # Add bounding boxes
                for j, box in enumerate(result['boxes']):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = result['class_ids'][j] if j < len(result['class_ids']) else 0
                    score = result['scores'][j] if j < len(result['scores']) else 0.0
                    
                    # Get class name safely
                    if result['class_names'] and class_id < len(result['class_names']):
                        class_name = result['class_names'][class_id]
                    else:
                        class_name = f"Class_{class_id}"
                    
                    label = f"{class_name}: {score:.2f}"
                    
                    # Use color in 0-1 range for bounding box
                    bbox_color = colors_01[j % len(colors_01)]
                    
                    # Add rectangle
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             fill=False, edgecolor=bbox_color, linewidth=2))
                    
                    # Add text label with proper background
                    ax.text(x1, y1 - 10, label, color='white', fontsize=8,
                           bbox=dict(facecolor=bbox_color, edgecolor='none', alpha=0.8, 
                                   boxstyle='round,pad=0.3'))

                # Add information text
                info_text = (f"Objects: {result['num_objects']}\n"
                           f"Area: {result['total_mask_area_px']:.0f} px\n"
                           f"Coverage: {result['normalized_mask_area_percent']:.2f}%\n"
                           f"Fuzzy: {result['fuzzy_area_classification']}")
                ax.set_title(f"Image {i+1}\n{info_text}", fontsize=10, fontweight='bold')

            # Remove unused subplots
            for j in range(num_images, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Save the figure
            plot_filename = f"inference_results_{title.replace(' ', '_').lower()}.png"
            plot_path = os.path.join(self.INFERENCE_OUTPUT_DIR, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='black', format='png')
            print(f"📊 High-quality plot saved: {plot_path}")
            
            # Also save as PDF for academic papers
            pdf_path = plot_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='black', format='pdf')
            print(f"📄 PDF version saved: {pdf_path}")
            
            if not save_only:
                plt.show()
            else:
                plt.close(fig)  # Clean up memory
                
            return plot_path
            
        except Exception as e:
            print(f"❌ Error in visualization: {str(e)}")
            print(f"   Falling back to simple export mode...")
            
            # Fallback: Save individual images with OpenCV annotations
            return self._fallback_visualization(inference_results, title)
            
        finally:
            plt.close('all')  # Clean up any remaining figures

    def _fallback_visualization(self, inference_results, title):
        """
        Fallback visualization method using OpenCV when matplotlib fails.
        Saves individual annotated images.
        """
        output_folder = os.path.join(self.INFERENCE_OUTPUT_DIR, f"fallback_{title.replace(' ', '_').lower()}")
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"🔄 Using fallback visualization, saving to: {output_folder}")
        
        for i, result in enumerate(inference_results):
            img = cv2.imread(result['image_path'])
            if img is None:
                continue
                
            # Define colors for OpenCV (BGR format)
            colors_bgr = [
                (0, 0, 255),    # Red
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (255, 255, 0),  # Cyan
                (0, 165, 255),  # Orange
                (255, 0, 128),  # Purple
                (0, 128, 0),    # Dark Green
                (128, 128, 128) # Gray
            ]
            
            # Draw masks
            if result['masks'] is not None and len(result['masks']) > 0:
                overlay = img.copy()
                for j, mask_data in enumerate(result['masks']):
                    mask_resized = cv2.resize(mask_data.astype(np.uint8), 
                                            (img.shape[1], img.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    color = colors_bgr[j % len(colors_bgr)]
                    overlay[mask_resized > 0] = color
                
                # Blend original with overlay
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
            
            # Draw bounding boxes and labels
            for j, box in enumerate(result['boxes']):
                x1, y1, x2, y2 = map(int, box)
                class_id = result['class_ids'][j] if j < len(result['class_ids']) else 0
                score = result['scores'][j] if j < len(result['scores']) else 0.0
                
                if result['class_names'] and class_id < len(result['class_names']):
                    class_name = result['class_names'][class_id]
                else:
                    class_name = f"Class_{class_id}"
                
                label = f"{class_name}: {score:.2f}"
                color = colors_bgr[j % len(colors_bgr)]
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add info text
            info_lines = [
                f"Objects: {result['num_objects']}",
                f"Area: {result['total_mask_area_px']:.0f} px",
                f"Coverage: {result['normalized_mask_area_percent']:.2f}%",
                f"Fuzzy: {result['fuzzy_area_classification']}"
            ]
            
            for idx, line in enumerate(info_lines):
                y_pos = 30 + idx * 25
                cv2.rectangle(img, (10, y_pos - 20), (300, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(img, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save individual image
            output_filename = f"image_{i+1}_{os.path.basename(result['image_path'])}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, img)
            print(f"  💾 Saved: {output_filename}")
        
        return output_folder

    def create_publication_ready_figures(self, inference_results, model_version, title="Inference Results"):
        """
        Create publication-ready figures specifically for academic papers.
        Generates multiple formats and layouts optimized for academic use.
        """
        if not inference_results:
            print("No inference results for publication figures.")
            return []
        
        pub_output_dir = os.path.join(self.INFERENCE_OUTPUT_DIR, f"publication_{model_version}")
        os.makedirs(pub_output_dir, exist_ok=True)
        
        generated_files = []
        
        # 1. Grid layout for overview
        try:
            grid_path = self.visualize_inference_results_grid(inference_results, 
                                                            f"{title} - {model_version.upper()}", 
                                                            save_only=True)
            if grid_path:
                generated_files.append(grid_path)
        except Exception as e:
            print(f"❌ Grid visualization failed: {e}")
        
        # 2. Individual high-quality images
        for i, result in enumerate(inference_results):
            try:
                individual_path = self._create_individual_publication_figure(result, i, model_version, pub_output_dir)
                if individual_path:
                    generated_files.append(individual_path)
            except Exception as e:
                print(f"❌ Individual figure {i} failed: {e}")
        
        # 3. Comparison figure (before/after)
        try:
            comparison_path = self._create_before_after_comparison(inference_results, model_version, pub_output_dir)
            if comparison_path:
                generated_files.append(comparison_path)
        except Exception as e:
            print(f"❌ Comparison figure failed: {e}")
        
        # 4. Results summary table
        try:
            table_path = self._create_results_summary_table(inference_results, model_version, pub_output_dir)
            if table_path:
                generated_files.append(table_path)
        except Exception as e:
            print(f"❌ Summary table failed: {e}")
        
        print(f"📚 Publication figures generated: {len(generated_files)} files")
        for file_path in generated_files:
            print(f"  📄 {file_path}")
        
        return generated_files

    def _create_individual_publication_figure(self, result, index, model_version, output_dir):
        """Create a single high-quality figure for individual image results."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
        
        # Load original image
        img = cv2.imread(result['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Left: Original image
        ax1.imshow(img_normalized)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Right: Annotated image
        overlay = np.zeros_like(img_normalized)
        colors_01 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        
        if result['masks'] is not None and len(result['masks']) > 0:
            for j, mask_data in enumerate(result['masks']):
                mask_resized = cv2.resize(mask_data.astype(np.uint8), 
                                        (img_rgb.shape[1], img_rgb.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                color = colors_01[j % len(colors_01)]
                mask_bool = mask_resized > 0
                overlay[mask_bool] = color
        
        blended = img_normalized * 0.6 + overlay * 0.4
        blended = np.clip(blended, 0, 1)
        ax2.imshow(blended)
        
        # Add bounding boxes
        for j, box in enumerate(result['boxes']):
            x1, y1, x2, y2 = map(int, box)
            class_id = result['class_ids'][j] if j < len(result['class_ids']) else 0
            score = result['scores'][j] if j < len(result['scores']) else 0.0
            
            if result['class_names'] and class_id < len(result['class_names']):
                class_name = result['class_names'][class_id]
            else:
                class_name = f"Class_{class_id}"
            
            label = f"{class_name}\n{score:.3f}"
            bbox_color = colors_01[j % len(colors_01)]
            
            ax2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=bbox_color, linewidth=2))
            ax2.text(x1, y1 - 5, label, color='white', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor=bbox_color, edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'))
        
        ax2.set_title(f'Detection Results\n{result["num_objects"]} objects, '
                     f'{result["normalized_mask_area_percent"]:.1f}% coverage\n'
                     f'Fuzzy: {result["fuzzy_area_classification"]}', 
                     fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_filename = f"individual_result_{index+1}_{model_version}"
        png_path = os.path.join(output_dir, f"{base_filename}.png")
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return png_path

    def _create_before_after_comparison(self, inference_results, model_version, output_dir):
        """Create a before/after comparison figure."""
        if len(inference_results) < 2:
            return None
            
        import matplotlib
        matplotlib.use('Agg')
        
        # Select 2 representative images
        selected_results = inference_results[:2]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        fig.suptitle(f'{model_version.upper()} Detection Results - Before/After Comparison', 
                    fontsize=16, fontweight='bold')
        
        for i, result in enumerate(selected_results):
            img = cv2.imread(result['image_path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            
            # Original (left column)
            axes[i, 0].imshow(img_normalized)
            axes[i, 0].set_title(f'Original Image {i+1}', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Annotated (right column)
            overlay = np.zeros_like(img_normalized)
            colors_01 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            
            if result['masks'] is not None and len(result['masks']) > 0:
                for j, mask_data in enumerate(result['masks']):
                    mask_resized = cv2.resize(mask_data.astype(np.uint8), 
                                            (img_rgb.shape[1], img_rgb.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    color = colors_01[j % len(colors_01)]
                    overlay[mask_resized > 0] = color
            
            blended = img_normalized * 0.6 + overlay * 0.4
            blended = np.clip(blended, 0, 1)
            axes[i, 1].imshow(blended)
            axes[i, 1].set_title(f'Detection Result {i+1}\n{result["num_objects"]} objects, '
                                f'{result["normalized_mask_area_percent"]:.1f}% coverage', 
                                fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        comparison_path = os.path.join(output_dir, f"before_after_comparison_{model_version}.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return comparison_path

    def _create_results_summary_table(self, inference_results, model_version, output_dir):
        """Create a summary table of results."""
        import matplotlib
        matplotlib.use('Agg')
        
        # Prepare data for table
        table_data = []
        for i, result in enumerate(inference_results):
            table_data.append([
                f"Image {i+1}",
                result['num_objects'],
                f"{result['total_mask_area_px']:.0f}",
                f"{result['normalized_mask_area_percent']:.2f}%",
                result['fuzzy_area_classification']
            ])
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.axis('tight')
        ax.axis('off')
        
        headers = ['Image', 'Objects', 'Area (px)', 'Coverage (%)', 'Fuzzy Class']
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternating row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f2f2f2')
        
        plt.title(f'{model_version.upper()} Detection Results Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        
        table_path = os.path.join(output_dir, f"results_summary_table_{model_version}.png")
        plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return table_path

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
        # Get compressed directory from config
        compressed_dir = getattr(self, 'config', {}).get('dataset', {}).get('default_compressed_dir', 'compressed')
        os.makedirs(compressed_dir, exist_ok=True)
        
        output_filename = os.path.join(compressed_dir, f"yolo{model_version}_segment_inference_results")
        if os.path.exists(source_folder):
            print(f"\n--- Mengkompresi folder hasil inferensi untuk YOLO{model_version} (segmentasi) ---")
            try:
                shutil.make_archive(output_filename, 'zip', source_folder)
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