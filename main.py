import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from ultralytics import YOLO

def extract_panels_ai(image_path, model_path='runs/detect/manga_panel_model/weights/best.pt'):
    """
    Mendeteksi panel manga menggunakan model AI YOLOv8.
    """
    if not os.path.exists(model_path):
        print("\n[PERINGATAN] Model AI khusus manga panel belum ditemukan!")
        print("Program akan menggunakan model YOLO default (yang mungkin akan mendeteksi objek acak seperti orang/kursi).")
        print("Silakan ikuti petunjuk untuk menjalankan 'train_ai.py' dengan dataset manga Anda terlebih dahulu.\n")
        model = YOLO('yolov8n.pt') # Fallback ke model dasar
    else:
        model = YOLO(model_path)
        
    print("AI sedang memindai gambar...")
    results = model(image_path)
    
    rects = []
    # Mengambil koordinat dari hasil deteksi AI
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # YOLO mengembalikan [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            # Anda bisa menambahkan filter 'confidence score' jika perlu
            # conf = box.conf[0].item()
            rects.append([int(x1), int(y1), int(w), int(h)])
            
    # Karena AI sudah cerdas membedakan panel dan objek di dalamnya,
    # kita tidak perlu lagi fungsi NMS (Non-Maximum Suppression) yang kompleks seperti sebelumnya.
    # Namun untuk keamanan, kita tetap saring sedikit jika ada kotak AI yang tumpang tindih berlebihan.
    def is_heavily_contained(r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        ix_min = max(x1, x2)
        iy_min = max(y1, y2)
        ix_max = min(x1+w1, x2+w2)
        iy_max = min(y1+h1, y2+h2)
        if ix_min < ix_max and iy_min < iy_max:
            inter_area = (ix_max - ix_min) * (iy_max - iy_min)
            area1 = w1 * h1
            if inter_area > 0.9 * area1: # Toleransi dinaikkan karena AI lebih presisi
                return True
        return False

    rects.sort(key=lambda r: r[2]*r[3], reverse=True)
    filtered_rects = []
    for r in rects:
        contained = False
        for fr in filtered_rects:
            if is_heavily_contained(r, fr):
                contained = True
                break
        if not contained:
            filtered_rects.append(r)

    # Baca gambar untuk digambar
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
    return filtered_rects, img

def sort_panels(rects):
    """
    Mengurutkan panel berdasarkan urutan baca manga:
    Dari Kanan ke Kiri, Atas ke Bawah.
    Menggunakan ukuran panel (tinggi) sebagai faktor toleransi.
    """
    remaining = list(rects)
    sorted_rects = []
    
    while remaining:
        # Cari titik Y paling atas
        min_y = min(r[1] for r in remaining)
        
        # Margin menggunakan tinggi rata-rata dari panel yang tersisa agar proporsional
        avg_h = sum(r[3] for r in remaining) / len(remaining)
        # Atau bisa pakai tinggi panel paling atas:
        top_panels = [r for r in remaining if r[1] == min_y]
        margin = top_panels[0][3] * 0.3 # Toleransi 30% dari tinggi panel paling atas
        
        row_candidates = [r for r in remaining if r[1] <= min_y + margin]
        
        # Pilih panel yang posisinya paling Kanan
        next_panel = max(row_candidates, key=lambda r: r[0])
        
        sorted_rects.append(next_panel)
        remaining.remove(next_panel)
        
    return sorted_rects

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Manga",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def main():
    print("Membuka dialog pemilihan file...")
    image_path = select_image()
    
    if not image_path or not os.path.exists(image_path):
        print("Tidak ada file yang dipilih atau file tidak ditemukan.")
        return

    print(f"File dipilih: {image_path}")
    print("Memulai pemindaian AI...")
    
    rects, img = extract_panels_ai(image_path)
    
    if not rects:
        print("Tidak ada objek yang terdeteksi.")
        return

    print(f"Berhasil mendeteksi {len(rects)} panel.")
    print("Mengurutkan panel sesuai alur baca manga...")
    
    sorted_rects = sort_panels(rects)
    
    img_draw = img.copy()
    
    print("\n--- Urutan Frame ---")
    for i, r in enumerate(sorted_rects):
        frame_number = i + 1
        x, y, w, h = r
        print(f"Frame {frame_number}: Posisi (X:{x}, Y:{y}, Lebar:{w}, Tinggi:{h})")
        
        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        cx, cy = x + 30, y + 30
        cv2.circle(img_draw, (cx, cy), 20, (0, 0, 255), -1)
        
        text_size = cv2.getTextSize(str(frame_number), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        cv2.putText(img_draw, str(frame_number), (tx, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    h_img, w_img = img_draw.shape[:2]
    max_height = 800
    if h_img > max_height:
        scale = max_height / h_img
        img_draw = cv2.resize(img_draw, (int(w_img * scale), max_height))
        
    print("\nTekan tombol apapun pada jendela gambar untuk menutup program.")
    cv2.imshow('MangTrans - Deteksi Frame (AI)', img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
