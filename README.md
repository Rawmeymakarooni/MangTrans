# MangTrans
Manga translation system that helps you learn japanese, an unerstan context. feel free to perfect the program on branches. At times i would merge some some programs into the main branch

## Panduan Pelatihan AI (YOLO) untuk Deteksi Panel Manga

Karena kita membuat AI khusus (Custom AI) untuk mengenali panel manga agar terhindar dari salah deteksi *chat bubble*, Anda perlu memberikan contoh gambar (data latih) kepada AI.

### Struktur Folder
Struktur folder dataset standar YOLO di dalam proyek Anda:
```text
MangTrans/
│
├── dataset/
│   ├── images/
│   │   ├── train/  <-- Taruh gambar manga untuk pelatihan di sini
│   │   └── val/    <-- Taruh sebagian kecil gambar untuk validasi di sini
│   └── labels/
│       ├── train/  <-- Taruh file anotasi (.txt) untuk pelatihan di sini
│       └── val/    <-- Taruh file anotasi (.txt) untuk validasi di sini
│
├── data.yaml       <-- File konfigurasi yang memberi tahu AI di mana letak datanya
├── train_ai.py     <-- Skrip untuk melatih AI
└── main.py         <-- Skrip utama (diperbarui untuk memakai AI)
```

### Langkah 1: Kumpulkan Gambar
1. Siapkan beberapa halaman manga (semakin banyak semakin baik, minimal 10-20 halaman untuk tes awal).
2. Simpan 80% dari gambar tersebut di folder `dataset/images/train/`.
3. Simpan 20% sisanya di folder `dataset/images/val/`.

### Langkah 2: Buat Label (Anotasi)
AI YOLO membutuhkan file teks (`.txt`) dengan nama yang persis sama dengan nama gambarnya (contoh: jika ada `manga_01.jpg`, maka harus ada `manga_01.txt`). File teks ini berisi koordinat *bounding box* dari setiap panel di halaman tersebut.

**Cara Termudah Membuat Anotasi:**
Sangat disarankan untuk **TIDAK** menulis koordinat secara manual. Gunakan platform gratis seperti **Roboflow** atau perangkat lunak lokal seperti **LabelImg**:
1. Buat proyek di [Roboflow](https://roboflow.com/) (Pilih tipe *Object Detection*).
2. Unggah gambar manga Anda.
3. Tarik kotak (*draw bounding box*) pada setiap panel (jangan tandai chat bubble). Beri label kotak tersebut dengan nama `panel`.
4. Setelah selesai, klik **Export Dataset** dan pilih format **YOLOv8**.
5. Ekstrak hasil unduhan tersebut dan letakkan isi folder `images/train` dan `labels/train` ke dalam folder `dataset/` di proyek kita. Lakukan hal yang sama untuk `val`.

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
