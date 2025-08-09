# file: modules/yolov8_visualizer_simple.py
import os
import json
import shutil
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import pandas as pd
from ultralytics import YOLO
from modules.fuzzy_area_classifier import FuzzyAreaClassifier

class YOLOv8Visualizer:
    """
    Simplified YOLOv8 visualizer that works with the existing inference system.
    Creates structured academic paper outputs.
    """
    
    def __init__(self, base_output_dir: str = "results/inference_outputs", img_size: tuple = (640, 640)):
        self.base_output_dir = base_output_dir
        self.img_size = img_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create YOLOv8 directory structure
        self.yolov8_dir = os.path.join(base_output_dir, "yolov8")
        self.metadata_dir = os.path.join(self.yolov8_dir, "metadata")
        self.publication_dir = os.path.join(self.yolov8_dir, "publication")
        self.figures_dir = os.path.join(self.publication_dir, "figures")
        self.individual_dir = os.path.join(self.publication_dir, "individual")
        self.markdown_dir = os.path.join(self.publication_dir, "markdown")
        self.annotated_dir = os.path.join(self.yolov8_dir, "annotated_images")
        self.compressed_dir = os.path.join(self.yolov8_dir, "compressed")
        
        # Create all directories
        for directory in [self.metadata_dir, self.figures_dir, self.individual_dir, 
                         self.markdown_dir, self.annotated_dir, self.compressed_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.fuzzy_classifier = FuzzyAreaClassifier()
        
        # Export settings
        self.export_settings = {
            "dpi": 300,
            "visualization_alpha": 0.4,
            "title_fontsize": 16,
            "subtitle_fontsize": 14,
            "label_fontsize": 12
        }
        
        # Colors in 0-1 range for matplotlib
        self.colors_01 = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0]
        ]
        
        print(f"🎯 YOLOv8Visualizer (Simple) initialized")
        print(f"📁 Output directory: {self.yolov8_dir}")

    def run_complete_visualization_pipeline(self, model_path: str, data_yaml_path: str, 
                                          num_images: int = 6, conf_threshold: float = 0.25,
                                          model_version: str = "v8n") -> Dict[str, Any]:
        """
        Run complete visualization pipeline using existing inference system.
        """
        print(f"🚀 Starting YOLOv8 Simple Visualization Pipeline")
        
        pipeline_results = {
            "status": "initialized",
            "model_info": {
                "model_path": model_path,
                "model_version": model_version,
                "confidence_threshold": conf_threshold,
                "num_images": num_images,
                "timestamp": self.timestamp
            },
            "generated_files": {
                "metadata": [],
                "figures": [],
                "individual": [],
                "annotated": [],
                "markdown": [],
                "compressed": []
            },
            "processing_log": []
        }
        
        try:
            # Step 1: Run inference using existing method
            self._log_step(pipeline_results, "Running inference on sample images")
            inference_results = self._run_inference_simplified(model_path, data_yaml_path, num_images, conf_threshold)
            
            if not inference_results:
                pipeline_results["status"] = "failed"
                pipeline_results["error"] = "No inference results generated"
                return pipeline_results
            
            # Step 2: Generate metadata
            self._log_step(pipeline_results, "Generating metadata files")
            metadata_files = self._generate_metadata_simplified(inference_results, model_version, conf_threshold)
            pipeline_results["generated_files"]["metadata"].extend(metadata_files)
            
            # Step 3: Create publication figures
            self._log_step(pipeline_results, "Creating publication figures")
            figure_files = self._create_publication_figures_simplified(inference_results, model_version)
            pipeline_results["generated_files"]["figures"].extend(figure_files)
            
            # Step 4: Generate markdown report
            self._log_step(pipeline_results, "Generating markdown report")
            markdown_files = self._generate_markdown_simplified(inference_results, model_version)
            pipeline_results["generated_files"]["markdown"].extend(markdown_files)
            
            # Step 5: Create compressed archive
            self._log_step(pipeline_results, "Creating compressed archive")
            compressed_files = self._create_archive_simplified(model_version)
            pipeline_results["generated_files"]["compressed"].extend(compressed_files)
            
            pipeline_results["status"] = "completed"
            total_files = sum(len(files) for files in pipeline_results['generated_files'].values())
            self._log_step(pipeline_results, f"Pipeline completed successfully - {total_files} files generated")
            
        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            self._log_step(pipeline_results, f"Pipeline failed: {str(e)}")
            print(f"❌ Pipeline failed: {e}")
        
        # Save pipeline results
        pipeline_file = os.path.join(self.metadata_dir, f"pipeline_results_{self.timestamp}.json")
        with open(pipeline_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        return pipeline_results

    def _run_inference_simplified(self, model_path: str, data_yaml_path: str, 
                                 num_images: int, conf_threshold: float) -> List[Dict]:
        """Simplified inference using existing pattern."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = YOLO(model_path)
        
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        dataset_root = data.get('path')
        test_path_in_yaml = data.get('test', 'test')
        class_names = data.get('names')
        
        # Find test images
        test_images_dir = os.path.join(dataset_root, test_path_in_yaml, 'images')
        if not os.path.exists(test_images_dir):
            test_images_dir = os.path.join(dataset_root, test_path_in_yaml)
        
        if not os.path.exists(test_images_dir):
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
        
        # Get sample images
        image_files = [
            os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {test_images_dir}")
        
        np.random.shuffle(image_files)
        sample_images = image_files[:num_images]
        
        print(f"📸 Processing {len(sample_images)} images")
        
        inference_results = []
        for i, img_path in enumerate(sample_images):
            try:
                original_img = cv2.imread(img_path)
                if original_img is None:
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
                            mask_resized = cv2.resize(mask.astype(np.uint8), (original_w, original_h), 
                                                    interpolation=cv2.INTER_NEAREST)
                            total_mask_area += np.sum(mask_resized > 0)
                        
                        if (original_w * original_h) > 0:
                            normalized_mask_area = (total_mask_area / (original_w * original_h)) * 100
                            fuzzy_area_classification = self.fuzzy_classifier.classify_area(normalized_mask_area)
                    
                    inference_results.append({
                        'image_id': f"img{i+1:03d}",
                        'image_path': img_path,
                        'image_filename': os.path.basename(img_path),
                        'boxes': boxes,
                        'scores': scores,
                        'class_ids': class_ids,
                        'masks': masks,
                        'num_objects': len(boxes),
                        'total_mask_area_px': total_mask_area,
                        'normalized_mask_area_percent': normalized_mask_area,
                        'fuzzy_area_classification': fuzzy_area_classification,
                        'class_names': class_names,
                        'original_width': original_w,
                        'original_height': original_h,
                        'confidence_scores': scores.tolist() if len(scores) > 0 else []
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        print(f"✅ Inference completed: {len(inference_results)} results")
        return inference_results

    def _generate_metadata_simplified(self, inference_results: List[Dict], model_version: str, 
                                    conf_threshold: float) -> List[str]:
        """Generate JSON and CSV metadata."""
        metadata_files = []
        
        # JSON metadata
        total_objects = sum(result['num_objects'] for result in inference_results)
        avg_confidence = np.mean([np.mean(result['confidence_scores']) for result in inference_results if result['confidence_scores']])
        
        metadata = {
            "experiment_info": {
                "model_version": model_version,
                "timestamp": self.timestamp,
                "confidence_threshold": conf_threshold,
                "total_images": len(inference_results),
                "total_objects": total_objects,
                "average_confidence": float(avg_confidence) if not np.isnan(avg_confidence) else 0.0
            },
            "image_results": inference_results
        }
        
        json_file = os.path.join(self.metadata_dir, f"inference_summary_{model_version}_{self.timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        metadata_files.append(json_file)
        
        # CSV summary
        csv_data = []
        for result in inference_results:
            csv_data.append({
                "Image_ID": result['image_id'],
                "Filename": result['image_filename'],
                "Objects_Detected": result['num_objects'],
                "Coverage_Percent": f"{result['normalized_mask_area_percent']:.2f}",
                "Fuzzy_Classification": result['fuzzy_area_classification']
            })
        
        csv_file = os.path.join(self.metadata_dir, f"inference_summary_{model_version}_{self.timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        metadata_files.append(csv_file)
        
        print(f"📊 Generated {len(metadata_files)} metadata files")
        return metadata_files

    def _create_publication_figures_simplified(self, inference_results: List[Dict], model_version: str) -> List[str]:
        """Create simplified publication figures."""
        figure_files = []
        
        try:
            # Create grid overview
            grid_file = self._create_grid_overview_simple(inference_results, model_version)
            if grid_file:
                figure_files.append(grid_file)
        except Exception as e:
            print(f"❌ Grid overview failed: {e}")
        
        print(f"📊 Generated {len(figure_files)} publication figures")
        return figure_files

    def _create_grid_overview_simple(self, inference_results: List[Dict], model_version: str) -> str:
        """Create a simple grid overview figure."""
        if not inference_results:
            return None
        
        num_images = len(inference_results)
        cols = min(3, num_images)
        rows = int(np.ceil(num_images / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), dpi=300, facecolor='white')
        
        # Handle single subplot case
        if num_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'YOLOv8 Detection Results - {model_version.upper()}', 
                    fontsize=16, fontweight='bold')
        
        for i, result in enumerate(inference_results):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Load image
            img = cv2.imread(result['image_path'])
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            
            # Create overlay
            overlay = np.zeros_like(img_normalized)
            
            if result['masks'] is not None and len(result['masks']) > 0:
                for j, mask_data in enumerate(result['masks']):
                    mask_resized = cv2.resize(mask_data.astype(np.uint8), 
                                            (img_rgb.shape[1], img_rgb.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    color = self.colors_01[j % len(self.colors_01)]
                    overlay[mask_resized > 0] = color
            
            # Blend
            alpha = self.export_settings["visualization_alpha"]
            blended = img_normalized * (1 - alpha) + overlay * alpha
            blended = np.clip(blended, 0, 1)
            
            ax.imshow(blended)
            ax.axis('off')
            
            # Add info
            info_text = (f"{result['image_id']}: {result['num_objects']} objects\n"
                        f"Coverage: {result['normalized_mask_area_percent']:.1f}%")
            ax.set_title(info_text, fontsize=12, fontweight='bold')
        
        # Remove unused subplots
        for j in range(num_images, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save
        png_path = os.path.join(self.figures_dir, f"grid_overview_yolov8_{num_images}images.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return png_path

    def _generate_markdown_simplified(self, inference_results: List[Dict], model_version: str) -> List[str]:
        """Generate simplified markdown report."""
        markdown_files = []
        
        total_objects = sum(result['num_objects'] for result in inference_results)
        avg_confidence = np.mean([np.mean(result['confidence_scores']) for result in inference_results if result['confidence_scores']])
        coverage_values = [result['normalized_mask_area_percent'] for result in inference_results]
        
        content = f"""# YOLOv8 Inference Results Report

## Experiment Details
- **Model**: {model_version.upper()} Instance Segmentation
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Images Processed**: {len(inference_results)}
- **Total Detections**: {total_objects} objects

## Results Summary
- **Average Confidence**: {avg_confidence:.1f}%
- **Coverage Range**: {min(coverage_values):.1f}% - {max(coverage_values):.1f}%

## Detailed Results
| Image ID | Objects | Coverage | Fuzzy Class |
|----------|---------|----------|-------------|
"""
        
        for result in inference_results:
            content += f"| {result['image_id']} | {result['num_objects']} | {result['normalized_mask_area_percent']:.1f}% | {result['fuzzy_area_classification']} |\n"
        
        content += f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        report_file = os.path.join(self.markdown_dir, f"results_report_{model_version}_{self.timestamp}.md")
        with open(report_file, 'w') as f:
            f.write(content)
        markdown_files.append(report_file)
        
        print(f"📝 Generated {len(markdown_files)} markdown reports")
        return markdown_files

    def _create_archive_simplified(self, model_version: str) -> List[str]:
        """Create compressed archive."""
        compressed_files = []
        
        try:
            archive_name = f"yolov8_{model_version}_results_{self.timestamp}"
            archive_path = os.path.join(self.compressed_dir, archive_name)
            shutil.make_archive(archive_path, 'zip', self.yolov8_dir)
            compressed_files.append(f"{archive_path}.zip")
            print(f"📦 Created archive: {archive_name}.zip")
        except Exception as e:
            print(f"❌ Archive creation failed: {e}")
        
        return compressed_files

    def _log_step(self, pipeline_results: Dict, message: str):
        """Log a pipeline step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        pipeline_results["processing_log"].append(log_entry)
        print(f"📝 {log_entry}")

