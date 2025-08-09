# Additional methods for YOLOv8Visualizer - to be merged with main file

def _create_compressed_archive(self, model_version: str) -> List[str]:
    """Create compressed archive of all outputs."""
    compressed_files = []
    
    try:
        archive_name = f"yolov8_{model_version}_inference_complete_{self.timestamp}"
        archive_path = os.path.join(self.compressed_dir, archive_name)
        
        # Create zip archive of the entire yolov8 directory
        shutil.make_archive(archive_path, 'zip', self.yolov8_dir)
        compressed_files.append(f"{archive_path}.zip")
        
        print(f"📦 Created compressed archive: {archive_name}.zip")
        
    except Exception as e:
        print(f"❌ Compression failed: {e}")
    
    return compressed_files

def get_export_summary(self) -> Dict[str, Any]:
    """Get a summary of export settings and structure."""
    return {
        "output_directory": self.yolov8_dir,
        "export_settings": self.export_settings,
        "figure_dimensions": self.figure_dimensions,
        "folder_structure": {
            "metadata": self.metadata_dir,
            "publication": self.publication_dir,
            "figures": self.figures_dir,
            "individual": self.individual_dir,
            "markdown": self.markdown_dir,
            "annotated": self.annotated_dir,
            "raw_outputs": self.raw_outputs_dir,
            "compressed": self.compressed_dir
        },
        "timestamp": self.timestamp
    }
