# Healthcare Data Preprocessing

Project preprocessing dataset kesehatan menggunakan **Python, Pandas, NumPy, Scikit-learn, dan FastAPI**.

Project ini dibuat untuk memenuhi kebutuhan preprocessing data pada mata kuliah Big Data, mulai dari pengecekan kualitas data, cleaning, transformation, hingga menghasilkan dataset yang siap digunakan untuk analisis dan visualisasi.

---

## Tema Dataset

**Klasifikasi Keluhan Kesehatan Berdasarkan Gejala Pasien**

Dataset merupakan dataset **sintetis/dibuat sendiri** yang berisi data pasien dan keluhan kesehatan.

Dataset memiliki:

- 1.000 baris data
- 14 atribut awal
- Data numerik
- Data kategorikal
- Data teks
- Target kategori keluhan
- Missing value
- Duplicate data
- Data tidak konsisten
- Outlier

---

## Struktur Project

```text
healthcare-preprocessing/
│
├── dataset/
│   └── dataset_keluhan_kesehatan_1000_baris_raw.csv
│
├── output/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── preprocessing.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md