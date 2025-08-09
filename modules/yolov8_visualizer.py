# file: modules/yolov8_visualizer.py
import os
import json
import csv
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
    Specialized visualizer for YOLOv8 with structured output for academic papers.
    Implements comprehensive folder structure, naming conventions, and export standards.
    """
    
    def __init__(self, base_output_dir: str = "results/inference_outputs", img_size: tuple = (640, 640)):
        self.base_output_dir = base_output_dir
        self.img_size = img_size
        self.model_name = "yolov8"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create structured directory hierarchy
        self.yolov8_dir = os.path.join(base_output_dir, "yolov8")
        self.metadata_dir = os.path.join(self.yolov8_dir, "metadata")
        self.publication_dir = os.path.join(self.yolov8_dir, "publication")
        self.figures_dir = os.path.join(self.publication_dir, "figures")
        self.individual_dir = os.path.join(self.publication_dir, "individual")
        self.markdown_dir = os.path.join(self.publication_dir, "markdown")
        self.annotated_dir = os.path.join(self.yolov8_dir, "annotated_images")
        self.raw_outputs_dir = os.path.join(self.yolov8_dir, "raw_outputs")
        self.compressed_dir = os.path.join(self.yolov8_dir, "compressed")
        
        # Create all directories
        for directory in [self.metadata_dir, self.figures_dir, self.individual_dir, 
                         self.markdown_dir, self.annotated_dir, self.raw_outputs_dir, 
                         self.compressed_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize fuzzy classifier
        self.fuzzy_classifier = FuzzyAreaClassifier()
        
        # Export settings
        self.export_settings = {
            "dpi": 300,
            "format_raster": "png",
            "format_vector": "pdf",
            "color_space": "RGB",
            "compression": "lossless",
            "visualization_alpha": 0.4,
            "font_family": "Arial",
            "title_fontsize": 16,
            "subtitle_fontsize": 14,
            "label_fontsize": 12,
            "caption_fontsize": 10
        }
        
        # Standard figure dimensions (inches)
        self.figure_dimensions = {
            "grid_layout": (16, 12),
            "individual_comparison": (12, 6),
            "summary_table": (10, 6),
            "before_after": (12, 10)
        }
        
        # Color scheme for consistency
        self.colors_01 = [
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
        
        print(f"🎯 YOLOv8Visualizer initialized")
        print(f"📁 Output directory: {self.yolov8_dir}")
        print(f"⚙️ Export settings: {self.export_settings['dpi']} DPI, {self.export_settings['format_raster'].upper()}/{self.export_settings['format_vector'].upper()}")

    def run_complete_visualization_pipeline(self, model_path: str, data_yaml_path: str, 
                                          num_images: int = 6, conf_threshold: float = 0.25,
                                          model_version: str = "v8n") -> Dict[str, Any]:
        """
        Run the complete visualization pipeline for YOLOv8.
        
        Args:
            model_path: Path to the trained YOLOv8 model
            data_yaml_path: Path to the dataset YAML file
            num_images: Number of test images to process
            conf_threshold: Confidence threshold for detections
            model_version: Specific YOLOv8 version (e.g., "v8n", "v8s")
        
        Returns:
            Dictionary containing all generated file paths and metadata
        """
        print(f"🚀 Starting YOLOv8 Complete Visualization Pipeline")
        print(f"📋 Model: {model_path}")
        print(f"🎯 Images: {num_images}, Confidence: {conf_threshold}")
        
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
            # Step 1: Run inference on sample images
            self._log_step(pipeline_results, "Running inference on sample images")
            inference_results = self._run_inference_on_sample_images(
                model_path, data_yaml_path, num_images, conf_threshold
            )
            
            if not inference_results:
                pipeline_results["status"] = "failed"
                pipeline_results["error"] = "No inference results generated"
                return pipeline_results
            
            # Step 2: Generate metadata
            self._log_step(pipeline_results, "Generating metadata files")
            metadata_files = self._generate_metadata(inference_results, model_version, conf_threshold)
            pipeline_results["generated_files"]["metadata"].extend(metadata_files)
            
            # Step 3: Create publication-ready figures
            self._log_step(pipeline_results, "Creating publication-ready figures")
            figure_files = self._create_publication_figures(inference_results, model_version)
            pipeline_results["generated_files"]["figures"].extend(figure_files)
            
            # Step 4: Generate individual comparison images
            self._log_step(pipeline_results, "Creating individual comparison images")
            individual_files = self._create_individual_comparisons(inference_results, model_version)
            pipeline_results["generated_files"]["individual"].extend(individual_files)
            
            # Step 5: Save annotated images
            self._log_step(pipeline_results, "Saving annotated images")
            annotated_files = self._save_annotated_images(inference_results, conf_threshold)
            pipeline_results["generated_files"]["annotated"].extend(annotated_files)
            
            # Step 6: Generate markdown reports
            self._log_step(pipeline_results, "Generating markdown reports")
            markdown_files = self._generate_markdown_reports(inference_results, model_version, pipeline_results)
            pipeline_results["generated_files"]["markdown"].extend(markdown_files)
            
            # Step 7: Create compressed archive
            self._log_step(pipeline_results, "Creating compressed archive")
            compressed_files = self._create_compressed_archive(model_version)
            pipeline_results["generated_files"]["compressed"].extend(compressed_files)
            
            pipeline_results["status"] = "completed"
            self._log_step(pipeline_results, f"Pipeline completed successfully - {sum(len(files) for files in pipeline_results['generated_files'].values())} files generated")
            
        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            self._log_step(pipeline_results, f"Pipeline failed: {str(e)}")
            print(f"❌ Pipeline failed: {e}")
        
        # Save pipeline results
        pipeline_file = os.path.join(self.metadata_dir, f"pipeline_results_{self.timestamp}.json")
        with open(pipeline_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        self._print_pipeline_summary(pipeline_results)
        return pipeline_results

    def _run_inference_on_sample_images(self, model_path: str, data_yaml_path: str, 
                                       num_images: int, conf_threshold: float) -> List[Dict]:
        """Run inference on sample images and collect results."""
        print(f"🔍 Running inference with model: {os.path.basename(model_path)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = YOLO(model_path)
        
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            import yaml
            data = yaml.safe_load(f)
        
        dataset_root = data.get('path')
        test_path_in_yaml = data.get('test')
        class_names = data.get('names')
        
        # Find test images directory
        test_images_dir = None
        possible_paths = [
            os.path.join(dataset_root, test_path_in_yaml, 'images'),
            os.path.join(dataset_root, test_path_in_yaml),
            os.path.join(dataset_root, 'test', 'images'),
            os.path.join(dataset_root, 'valid', 'images')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                test_images_dir = path
                break
        
        if not test_images_dir:
            raise FileNotFoundError(f"Test images directory not found. Tried: {possible_paths}")
        
        # Get sample images
        image_files = [
            os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {test_images_dir}")
        
        np.random.shuffle(image_files)
        sample_images = image_files[:num_images]
        
        print(f"📸 Processing {len(sample_images)} images from {test_images_dir}")
        
        inference_results = []
        for i, img_path in enumerate(sample_images):
            print(f"  {i+1}/{len(sample_images)}: {os.path.basename(img_path)}")
            
            # Load original image
            original_img = cv2.imread(img_path)
            if original_img is None:
                print(f"⚠️ Warning: Could not load {img_path}")
                continue
            
            original_h, original_w, _ = original_img.shape
            
            # Run inference
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
                
                num_objects = len(boxes)
                
                inference_results.append({
                    'image_id': f"img{i+1:03d}",
                    'image_path': img_path,
                    'image_filename': os.path.basename(img_path),
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
                    'original_height': original_h,
                    'confidence_scores': scores.tolist() if len(scores) > 0 else [],
                    'class_distribution': {class_names[cid]: int(np.sum(class_ids == cid)) for cid in np.unique(class_ids)} if len(class_ids) > 0 and class_names else {}
                })
        
        print(f"✅ Inference completed: {len(inference_results)} results")
        return inference_results

    def _generate_metadata(self, inference_results: List[Dict], model_version: str, 
                          conf_threshold: float) -> List[str]:
        """Generate comprehensive metadata files."""
        metadata_files = []
        
        # Calculate summary statistics
        total_objects = sum(result['num_objects'] for result in inference_results)
        avg_confidence = np.mean([np.mean(result['confidence_scores']) for result in inference_results if result['confidence_scores']])
        coverage_values = [result['normalized_mask_area_percent'] for result in inference_results]
        
        fuzzy_dist = {}
        for result in inference_results:
            fuzzy_class = result['fuzzy_area_classification']
            fuzzy_dist[fuzzy_class] = fuzzy_dist.get(fuzzy_class, 0) + 1
        
        # JSON metadata
        metadata = {
            "experiment_info": {
                "model_version": model_version,
                "timestamp": self.timestamp,
                "confidence_threshold": conf_threshold,
                "image_size": self.img_size
            },
            "processing_settings": {
                "num_inference_images": len(inference_results),
                "export_dpi": self.export_settings["dpi"],
                "color_space": self.export_settings["color_space"],
                "visualization_alpha": self.export_settings["visualization_alpha"]
            },
            "results_summary": {
                "total_images_processed": len(inference_results),
                "total_objects_detected": total_objects,
                "average_confidence": float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
                "coverage_range": f"{min(coverage_values):.1f}%-{max(coverage_values):.1f}%" if coverage_values else "0%-0%",
                "fuzzy_classifications": fuzzy_dist
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
                "Total_Area_px": result['total_mask_area_px'],
                "Coverage_Percent": f"{result['normalized_mask_area_percent']:.2f}",
                "Fuzzy_Classification": result['fuzzy_area_classification'],
                "Avg_Confidence": f"{np.mean(result['confidence_scores']):.3f}" if result['confidence_scores'] else "0.000"
            })
        
        csv_file = os.path.join(self.metadata_dir, f"inference_summary_{model_version}_{self.timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        metadata_files.append(csv_file)
        
        # Model info
        model_info = {
            "model_version": model_version,
            "export_timestamp": self.timestamp,
            "total_parameters": "N/A",  # Could be extracted from model
            "model_size_mb": "N/A",
            "inference_settings": {
                "confidence_threshold": conf_threshold,
                "image_size": self.img_size,
                "device": "auto"
            }
        }
        
        model_info_file = os.path.join(self.metadata_dir, f"model_info_{model_version}_{self.timestamp}.json")
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        metadata_files.append(model_info_file)
        
        print(f"📊 Generated {len(metadata_files)} metadata files")
        return metadata_files

    def _log_step(self, pipeline_results: Dict, message: str):
        """Log a pipeline step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        pipeline_results["processing_log"].append(log_entry)
        print(f"📝 {log_entry}")

    def _print_pipeline_summary(self, pipeline_results: Dict):
        """Print a comprehensive pipeline summary."""
        print(f"\n{'='*80}")
        print(f"🎯 YOLOv8 VISUALIZATION PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Status: {pipeline_results['status'].upper()}")
        print(f"⏰ Timestamp: {pipeline_results['model_info']['timestamp']}")
        print(f"🏷️ Model: {pipeline_results['model_info']['model_version']}")
        print(f"📸 Images Processed: {pipeline_results['model_info']['num_images']}")
        
        total_files = sum(len(files) for files in pipeline_results['generated_files'].values())
        print(f"📁 Total Files Generated: {total_files}")
        
        for category, files in pipeline_results['generated_files'].items():
            if files:
                print(f"  📂 {category.title()}: {len(files)} files")
        
        print(f"📁 Output Directory: {self.yolov8_dir}")
        print(f"{'='*80}\n")

    # Additional methods for creating figures, individual comparisons, etc. would go here
    # (Implementation continues with _create_publication_figures, _create_individual_comparisons, etc.)
