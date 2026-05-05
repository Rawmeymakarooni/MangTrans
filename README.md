# MangTrans
Manga translation system that helps you learn japanese, an unerstan context. feel free to perfect the program on branches. At times i would merge some some programs into the main branch

## Panduan Pelatihan AI (YOLO) untuk Deteksi Panel Manga

Karena kita membuat AI khusus (Custom AI) untuk mengenali panel manga agar terhindar dari salah deteksi *chat bubble*, Anda perlu memberikan contoh gambar (data latih) kepada AI.

### Struktur Folder (Standar Roboflow Export)
Jika Anda mengunduh (export) dataset dari Roboflow menggunakan format **YOLOv8**, Anda akan mendapatkan folder (misalnya `manga panels.v1i.yolov8`). Saya telah mengubah nama folder tersebut menjadi `dataset` dan menyesuaikan strukturnya.

Strukturnya sekarang terlihat seperti ini:
```text
MangTrans/
│
├── dataset/            <-- Ini adalah folder hasil unduhan Roboflow yang di-rename
│   ├── train/          <-- Data utama untuk melatih AI
│   │   ├── images/     <-- Berisi file gambar (.jpg)
│   │   └── labels/     <-- Berisi file anotasi (.txt)
│   ├── valid/          <-- Data untuk mengevaluasi akurasi AI selama pelatihan
│   │   ├── images/
│   │   └── labels/
│   ├── test/           <-- (Opsional) Data tes tambahan
│   └── data.yaml       <-- (Abaikan file ini, kita pakai data.yaml di folder utama)
│
├── data.yaml       <-- File konfigurasi utama yang memberi tahu AI letak foldernya
├── train_ai.py     <-- Skrip untuk melatih AI
└── main.py         <-- Skrip utama
```

### Cara Mengupdate Data Baru di Kemudian Hari
Jika nanti Anda membuat data baru di Roboflow dan mengunduhnya lagi:
1. Hapus folder `dataset` yang lama.
2. Ekstrak folder hasil unduhan Roboflow yang baru ke dalam proyek.
3. Ubah nama folder tersebut (misalnya dari `manga panels.v1i.yolov8`) menjadi `dataset`.
4. Anda tidak perlu merombak isinya, langsung saja jalankan langkah pelatihan di bawah.

### Langkah 3: Jalankan Pelatihan AI
Setelah folder `dataset/images` dan `dataset/labels` Anda terisi:
1. Buka terminal.
2. Jalankan skrip pelatihan:
   ```bash
   python train_ai.py
   ```
3. Proses ini akan memakan waktu. Setelah selesai, model AI yang sudah pintar akan disimpan secara otomatis di direktori `runs/detect/manga_panel_model/weights/best.pt`.

### Langkah 4: Tes Hasilnya
Jalankan skrip utama:
```bash
python main.py
```
Skrip ini sekarang akan secara otomatis mencari model `best.pt` hasil pelatihan Anda dan menggunakannya untuk mendeteksi panel manga dengan sangat akurat!
