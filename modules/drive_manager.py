# file: modules/drive_manager.py
import os
import shutil

class DriveManager:
    """
    Mengelola penyimpanan file hasil eksperimen ke Google Drive.
    """
    def __init__(self, colab_root_path='/content/', drive_base_path='/content/gdrive/MyDrive/'):
        self.colab_root = colab_root_path
        self.drive_root = drive_base_path
        
    def save_all_results_to_drive(self, folder_to_save="yolo_waste_results"):
        """
        Menyimpan folder yang berisi semua hasil (zips, plots) dari lingkungan Colab
        ke Google Drive.
        
        Args:
            folder_to_save (str): Nama folder tujuan di Google Drive.
        """
        source_dir = self.colab_root
        destination_dir = os.path.join(self.drive_root, folder_to_save)

        if not os.path.exists(destination_dir):
            print(f"Membuat folder tujuan di Google Drive: {destination_dir}")
            os.makedirs(destination_dir)
        else:
            print(f"Folder tujuan sudah ada: {destination_dir}. Menghapus konten lama...")
            try:
                # Menghapus konten lama untuk menghindari duplikasi
                for item in os.listdir(destination_dir):
                    item_path = os.path.join(destination_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            except Exception as e:
                print(f"Peringatan: Gagal menghapus konten lama: {e}. Melanjutkan...")
        
        print(f"\n--- Menyimpan semua hasil ke Google Drive ---")

        # Daftar file dan folder yang ingin disimpan
        items_to_save = ['datasets.zip', 'rknn_models.zip', 'results/onnx_models', 'results/runs']
        
        for item in items_to_save:
            source_path = os.path.join(source_dir, item)
            dest_path = os.path.join(destination_dir, item)
            
            if os.path.exists(source_path):
                try:
                    if os.path.isdir(source_path):
                        print(f"Menyalin folder: {item}...")
                        shutil.copytree(source_path, dest_path)
                    else: # File
                        print(f"Menyalin file: {item}...")
                        shutil.copy2(source_path, dest_path)
                    print(f"Berhasil menyalin {item}.")
                except Exception as e:
                    print(f"Gagal menyalin {item}: {e}")
            else:
                print(f"Peringatan: File atau folder '{item}' tidak ditemukan di '{source_path}'.")

        print("\nSelesai. Semua file hasil telah disimpan ke Google Drive.")
