from ultralytics import YOLO
import os

def main():
    # Pastikan data.yaml ada
    if not os.path.exists('data.yaml'):
        print("Error: File data.yaml tidak ditemukan!")
        return

    print("=== Memulai Proses Pelatihan AI Manga Panel ===")
    print("Mendownload model YOLOv8n (Nano) pre-trained sebagai dasar...")
    
    # Memuat model YOLO dasar (yolov8n.pt akan otomatis didownload jika belum ada)
    model = YOLO('yolov8n.pt')

    print("Memulai proses fine-tuning dengan dataset Anda...")
    # Anda dapat mengubah parameter di bawah ini sesuai spesifikasi komputer
    # Jika menggunakan GPU, akan jauh lebih cepat. Jika CPU, biarkan 'epochs' tetap kecil untuk tes.
    results = model.train(
        data='data.yaml',
        epochs=50,       # Jumlah putaran pelatihan (bisa ditambah jadi 100-300 untuk hasil maksimal)
        imgsz=640,       # Ukuran gambar pelatihan
        batch=8,         # Ukuran batch (turunkan jika memori tidak cukup)
        name='manga_panel_model' # Nama folder penyimpanan hasil (di dalam folder 'runs/detect/')
    )
    
    print("\nPelatihan Selesai!")
    print("Model terbaik Anda tersimpan di: runs/detect/manga_panel_model/weights/best.pt")

if __name__ == '__main__':
    main()
