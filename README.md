# Laporan Proyek Machine Learning - Azel Fabian Azmi

Proyek ini bertujuan membangun model machine learning untuk memprediksi harga mobil berdasarkan berbagai fitur kendaraan seperti tahun, transmisi, bahan bakar, jarak tempuh, warna, dan lainnya.

---

## Project Domain

Industri otomotif khususnya pasar mobil bekas mengalami perkembangan yang sangat dinamis. Penentuan harga kendaraan bekas merupakan proses kompleks yang dipengaruhi oleh berbagai faktor seperti tahun pembuatan, jarak tempuh, tipe mesin, bahan bakar, hingga warna kendaraan. Penjual sering kali kesulitan menentukan harga optimal karena adanya ketidaksesuaian antara kondisi mobil dengan harga pasar. Di sisi lain, pembeli juga kesulitan membandingkan harga secara objektif antar kendaraan yang serupa.

Dengan pendekatan machine learning, estimasi harga dapat dilakukan berdasarkan pola historis dan hubungan antar fitur. Proyek ini menggunakan algoritma regresi seperti Linear Regression, K-Nearest Neighbors, Decision Tree, dan Random Forest untuk membangun model prediksi harga kendaraan. Model ini diharapkan dapat digunakan dalam platform jual beli mobil bekas maupun oleh individu untuk memperoleh harga estimasi yang lebih adil dan akurat.

### Bagaimana Masalah tersebut Diselesaikan?

Masalah prediksi harga kendaraan diselesaikan melalui pendekatan terstruktur dalam beberapa tahapan berikut:

#### 1. Eksplorasi dan Pemahaman Data (Data Understanding)
Data dieksplorasi untuk mengetahui struktur, tipe fitur, nilai ekstrem, serta distribusi data. Tahap ini juga dilakukan visualisasi univariat dan bivariat untuk memahami pola dan korelasi antar variabel terhadap harga.

#### 2. Pembersihan dan Persiapan Data (Data Preparation)
Tahapan ini mencakup:
- Menghapus data duplikat
- Menangani nilai hilang (missing value) dengan metode pengisian sesuai tipe data
- Encoding fitur kategorikal agar dapat diterima oleh model
- Normalisasi dan transformasi fitur jika diperlukan

#### 3. Pemilihan Fitur dan Target
Memisahkan variabel target (price) dan fitur-fitur prediktor (year, mileage, fuel, dll.). Fitur yang tidak relevan seperti description dihapus untuk meningkatkan kualitas model.

#### 4. Pemodelan dengan Berbagai Algoritma Regresi
Beberapa model machine learning digunakan untuk membandingkan performa prediksi harga:
- Linear Regression sebagai baseline
- K-Nearest Neighbors (KNN) untuk pendekatan berbasis kemiripan
- Decision Tree untuk pemetaan keputusan secara hierarki
- Random Forest sebagai model ensemble untuk hasil prediksi yang lebih stabil dan akurat

#### 5. Evaluasi Model
Setiap model diuji menggunakan metrik MAE, RMSE, dan R² Score. Model terbaik dipilih berdasarkan kombinasi error terkecil dan skor prediksi tertinggi.

Melalui pendekatan ini, sistem prediksi harga kendaraan dapat dibangun secara efisien dan dapat memberikan estimasi yang mendekati kondisi pasar aktual.


### State Of The Art Penelitian Sebelumnya

Dalam bidang prediksi harga kendaraan, pendekatan regresi berbasis machine learning telah menjadi metode populer karena kemampuannya menangani data kompleks dan beragam fitur.

Penelitian oleh Gupta et al. [1] menunjukkan bahwa Random Forest memberikan performa yang unggul dalam hal akurasi dan kestabilan dalam memprediksi harga kendaraan dibandingkan algoritma lain. Random Forest dapat mengurangi risiko overfitting yang umum terjadi pada Decision Tree dengan menggabungkan prediksi dari banyak pohon secara acak.

Sementara itu, penelitian lain oleh Singh dan Kumari [2] membandingkan performa Linear Regression, KNN, dan Decision Tree dalam prediksi harga kendaraan. Hasil studi tersebut menunjukkan bahwa model ensemble seperti Random Forest dan Gradient Boosting lebih unggul dalam menangkap hubungan non-linear antar fitur, sedangkan model sederhana seperti Linear Regression cenderung underfitting ketika fitur memiliki distribusi kompleks.

Penggunaan kombinasi beberapa algoritma regresi dalam proyek ini mengikuti arah yang sejalan dengan penelitian terkini, dengan fokus pada perbandingan performa serta pemilihan model terbaik berdasarkan metrik evaluasi.

---

## Business Understanding

Harga kendaraan sering kali menjadi salah satu faktor utama yang mempengaruhi keputusan pembelian atau penjualan. Namun, menentukan harga kendaraan yang wajar dapat menjadi tantangan, terutama ketika mempertimbangkan berbagai faktor yang memengaruhi harga kendaraan, seperti tahun pembuatan, merek, tipe mesin, jarak tempuh (mileage), dan banyak atribut teknis lainnya. Banyak pembeli dan penjual yang kesulitan dalam mendapatkan harga yang sesuai dengan kondisi kendaraan, karena tidak memiliki referensi harga yang tepat berdasarkan data yang ada di pasar. 

Sebagian besar transaksi jual beli kendaraan masih mengandalkan penilaian subyektif atau informasi pasar yang terbatas, yang dapat menyebabkan ketidaksesuaian harga, baik terlalu mahal atau terlalu murah. Hal ini menciptakan ketidakpastian dan ketidakseimbangan informasi antara penjual dan pembeli. Sebagai contoh, kendaraan yang baru diproduksi dengan fitur premium mungkin dijual dengan harga yang tidak proporsional, sementara kendaraan bekas dengan kondisi baik mungkin dihargai lebih rendah dari nilai pasar sebenarnya. 

Masalah lainnya adalah adanya perbedaan harga yang besar antara kendaraan dengan spesifikasi yang hampir sama, hanya karena perbedaan kecil dalam faktor seperti warna eksterior, kondisi mesin, atau lokasi penjualan. Ini menjadi tantangan bagi calon pembeli yang ingin memastikan bahwa mereka mendapatkan harga yang adil sesuai dengan kondisi dan kualitas kendaraan yang mereka beli.

### Problem Statements
- Bagaimana memprediksi harga mobil bekas secara akurat berdasarkan fitur-fitur kendaraan?
- Model machine learning regresi mana yang memiliki performa terbaik dalam memprediksi harga kendaraan?

### Goals
- Membangun model prediksi harga kendaraan menggunakan beberapa algoritma regresi.
- Mengevaluasi dan membandingkan performa setiap model menggunakan metrik regresi.

### Solution Statements
- Menggunakan empat algoritma regresi: Linear Regression, KNN, Decision Tree, dan Random Forest.
- Memilih model terbaik berdasarkan metrik MAE, RMSE, dan R² Score.

---

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | _Vehicle Price Prediction_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/khwaishsaxena/vehicle-price-prediction-dataset/data) |
| Maintainer | [Khwaish Saxena](https://www.kaggle.com/khwaishsaxena) |
| License | Community Data License Agreement – Sharing, Version 1.0 |
| Visibility | Publik |
| Tags | _Business, Automobiles and Vehicles, Regression_ |
| View | 448 |


Tabel 1. Informasi Dataset

![data info](https://github.com/user-attachments/assets/99ab422f-f692-40f8-85a0-8b33762b59da)


Dilihat dari _Tabel 1. Informasi Dataset_ dataset ini berisi informasi sebagai berikut ini : 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 1002 sample dengan 17 fitur.

### Variable - variable pada dataset
- name (categorical): Nama lengkap kendaraan, mencakup tahun, model, dan trim.
- description (categorical): Deskripsi singkat tentang kendaraan, termasuk fitur-fitur tambahan atau informasi terkait kendaraan.
- make (categorical): Merek atau produsen kendaraan.
- model (categorical): Model kendaraan, yang sering kali merupakan varian dari merek yang sama.
- year (numeric): Tahun pembuatan kendaraan.
- price  (numeric): Harga kendaraan dalam satuan dolar.
- engine (categorical): Jenis dan spesifikasi mesin kendaraan, termasuk kapasitas dan tipe.
- cylinders (numeric): Jumlah silinder mesin kendaraan.
- fuel (categorical): Jenis bahan bakar yang digunakan oleh kendaraan.
- mileage (numeric): Jarak tempuh kendaraan dalam satuan mil.
- transmission (categorical): Jenis transmisi kendaraan.
- trim (categorical): Varian atau trim dari kendaraan, yang mencakup fitur-fitur tertentu.
- body (categorical): Jenis bodi kendaraan, seperti SUV, Pickup Truck, dan lain-lain.
- doors (numeric): Jumlah pintu kendaraan.
- exterior_color (categorical): Warna eksterior kendaraan.
- interior_color (categorical): Warna interior kendaraan.
- drivetrain (categorical): Tipe drivetrain kendaraan, seperti penggerak empat roda (4WD) atau penggerak dua roda (FWD).

### Informasi Dataset

![EDA Describe Data](https://github.com/user-attachments/assets/1fb877bd-18b8-499b-8db8-962e233df494)

Gambar 2. Penjelasan Dataset

Gambar 2 merupakan penjelasan mengenai dataset yang digunakan
- **Kolom `year`** mencatat tahun pembuatan kendaraan. Rata-rata tahun pembuatan kendaraan adalah 2023, dengan sebagian besar kendaraan dibuat pada tahun 2024. Rentang tahun berada antara 2023 hingga 2025, yang menunjukkan bahwa dataset ini sebagian besar terdiri dari kendaraan terbaru.
- **Kolom `price`** mencatat harga kendaraan dalam dolar. Rata-rata harga kendaraan adalah $50,202.99, dengan harga minimum mencapai $0, yang bisa jadi disebabkan oleh data yang hilang atau salah input. Harga maksimum mencapai $195,895, yang menunjukkan adanya kendaraan dengan harga jauh lebih tinggi, mungkin karena model premium atau kendaraan khusus. Sebagian besar harga berada di kisaran $36,900 hingga $58,717, mencerminkan harga kendaraan yang lebih umum di pasar.
- **Kolom `cylinders`** mencatat jumlah silinder pada mesin kendaraan. Rata-rata jumlah silinder adalah 4.98, dengan sebagian besar kendaraan memiliki 4 atau 6 silinder. Ada juga beberapa kendaraan yang memiliki 8 silinder, meskipun jumlahnya relatif sedikit. Nilai minimum untuk jumlah silinder adalah 0, yang kemungkinan disebabkan oleh data yang tidak valid.
- **Kolom `mileage`** mencatat jarak tempuh kendaraan dalam mil. Rata-rata jarak tempuh adalah 69 mil, dengan sebagian besar kendaraan memiliki jarak tempuh yang rendah, berada di bawah 14 mil. Namun, ada kendaraan dengan jarak tempuh ekstrem mencapai 9,711 mil, yang bisa jadi merupakan outlier atau kendaraan dengan penggunaan intensif. Rentang jarak tempuh yang luas menunjukkan variasi besar dalam data.
- **Kolom `doors`** mencatat jumlah pintu kendaraan. Rata-rata jumlah pintu adalah 3.94, dengan sebagian besar kendaraan memiliki 4 pintu. Rentang nilai untuk jumlah pintu berada antara 2 hingga 5 pintu, menunjukkan bahwa kebanyakan kendaraan dalam dataset ini adalah sedan atau SUV yang umumnya memiliki 4 pintu.


### Pengecekan Data Duplikat dan Missing Value
-	Data Duplikat

![Data Duplikat](https://github.com/user-attachments/assets/9e166ea6-33d5-4cbc-b838-ee83b7b10b60)

Gambar 3. Data Duplikat.

Pada gambar tersebut, menjelaskan bahwa pada dataset ini memiliki 24 data yang terduplikat.

-	Missing Value
  
![Missing Value](https://github.com/user-attachments/assets/d9506df0-7468-4555-ad9e-6f8154775519)

Gambar 4. Missing Value

Pada gambar tersebut, menjelaskan bahwa pada dataset ini memiliki banyak missing value.

Dengan adanya data terduplikat dan missing value, maka dilakukannya pengisian/penghapusan terhadap nilai tersebut untuk data menjadi bersih.

![Data Bersih](https://github.com/user-attachments/assets/a38f12fd-864f-4144-891c-0f76a21341e6)

Gambar 5. Data Bersih

### Pengecekan Value Unik yang Ada Pada Dataset

![Value Unik](https://github.com/user-attachments/assets/209cb64c-37e9-4e7d-a26d-e48f8162a84f)

Gambar 6. Value Unik

Berdasarkan gambar di atas, berikut adalah deskripsi singkat mengenai banyaknya nilai unik dari masing-masing fitur dalam dataset:
- price (860 unique values)
Menunjukkan variasi harga kendaraan, dengan 860 nilai unik yang mencerminkan spektrum harga yang luas. Ini menunjukkan bahwa hampir setiap kendaraan memiliki harga berbeda, menandakan keberagaman pasar mobil bekas.
- description (762 unique values)
Merupakan deskripsi bebas yang dituliskan untuk kendaraan. Nilai unik yang tinggi (762) menandakan bahwa deskripsi ini sangat bervariasi dan bersifat unik per entri. Fitur ini sulit digunakan dalam model kecuali diolah lebih lanjut (misalnya dengan NLP).
- name (358 unique values)
Menyatakan nama spesifik kendaraan. Nilai unik yang cukup tinggi (358) menunjukkan beragamnya jenis kendaraan yang tersedia.
- exterior_color (263 unique values)
Menggambarkan variasi warna luar kendaraan. Dengan 263 nilai unik, terdapat banyak variasi penamaan atau kombinasi warna yang berbeda, kemungkinan termasuk deskripsi warna kustom.
- trim (197 unique values)
Merujuk pada tipe atau versi dari suatu model kendaraan. Banyaknya nilai unik menunjukkan variasi varian mobil dalam dataset.
- model (153 unique values)
Mewakili model kendaraan. Terdapat 153 model berbeda, yang mencerminkan keberagaman produk dalam dataset.
- engine (100 unique values)
Menunjukkan variasi kapasitas mesin atau jenis mesin. Nilai ini mengindikasikan keberagaman dalam performa mesin yang ditawarkan.
- mileage (96 unique values)
Menyatakan jarak tempuh kendaraan. 96 nilai unik menunjukkan variasi kondisi penggunaan kendaraan bekas.
- interior_color (91 unique values)
Menggambarkan variasi warna interior mobil, yang dapat mempengaruhi preferensi konsumen terhadap estetika.
- transmission (38 unique values)
Menyatakan tipe transmisi kendaraan. Nilai ini tampaknya termasuk beberapa penamaan ulang atau variasi gaya penulisan (misalnya “Automatic” vs “Auto”).
- make (28 unique values)
Mewakili merek kendaraan. Terdapat 28 merek berbeda yang tercakup dalam dataset.
- body (8 unique values)
Merujuk pada bentuk bodi kendaraan seperti sedan, SUV, coupe, dll. Ini merupakan kategori yang cukup standar dalam dunia otomotif.
- fuel (7 unique values)
Menyatakan jenis bahan bakar seperti bensin, solar, listrik, hybrid, dll. 7 jenis bahan bakar menunjukkan adanya tren kendaraan energi alternatif juga.
- cylinders (6 unique values)
Mengindikasikan jumlah silinder pada mesin kendaraan. Variasi ini memengaruhi performa dan efisiensi bahan bakar.
- doors (5 unique values)
Menunjukkan jumlah pintu kendaraan (misalnya 2, 4, atau 5 pintu). Cukup konsisten dengan desain umum kendaraan.
- drivetrain (4 unique values)
Menggambarkan sistem penggerak seperti FWD, RWD, AWD, dan 4WD. Ini memengaruhi pengalaman berkendara dan performa di berbagai kondisi jalan.
- year (3 unique values)
Merupakan tahun produksi kendaraan. Nilai unik yang hanya 3 kemungkinan menunjukkan bahwa dataset hanya mencakup mobil dari tiga tahun produksi tertentu.

### EDA - Multivariate Analysis

![Gambar Corrmap](https://github.com/user-attachments/assets/ee986052-5724-4fb6-be36-063274aed226)

Gambar 7. Heatmap Korelasi

Heatmap pada gambar 7 ini menggambarkan korelasi antar fitur numerik seperti `year`, `price`, `cylinders`, `mileage`, dan `doors`. Temuan utama:
- Korelasi positif sedang antara `year` dan `price` menunjukkan bahwa kendaraan yang lebih baru cenderung memiliki harga lebih tinggi.
- Korelasi negatif antara `mileage` dan `price` menegaskan bahwa semakin tinggi jarak tempuh, biasanya harga kendaraan lebih rendah.
- Fitur seperti `cylinders` memiliki korelasi lemah terhadap harga, namun tetap bisa relevan untuk model prediksi.

![Price - Body v DT](https://github.com/user-attachments/assets/6cbaa169-46e5-4927-831a-1c887159f719)

Gambar 8. Harga berdasarkan Jenis Body dan Drivetrain

Gambar 8 menunjukkan distribusi harga kendaraan berdasarkan jenis body (`body`) dan drivetrain (`drivetrain`). Hasil pengamatan menunjukkan:
- **SUV** merupakan tipe body yang paling umum dengan variasi harga yang luas.
- **Pickup Truck** cenderung memiliki median harga yang tinggi, dan sering dikombinasikan dengan drivetrain `Four-wheel Drive`.
- Tipe drivetrain juga memberikan pengaruh yang signifikan terhadap harga, di mana kendaraan dengan `All-wheel Drive` atau `Four-wheel Drive` cenderung memiliki harga lebih tinggi dibandingkan `Front-wheel Drive`.

![Mil v Price - fuel](https://github.com/user-attachments/assets/5053b55e-995d-43fd-972d-cfe82899eb3a)

Gambar 9. Mileage vs Price berdasarkan Jenis Bahan Bakar

Gambar 9 menunjukkan hubungan antara `mileage` (jarak tempuh) dan `price` (harga kendaraan), dengan pewarnaan (`hue`) berdasarkan jenis bahan bakar (`fuel`). Dari grafik terlihat bahwa:
- Secara umum, harga kendaraan cenderung menurun seiring bertambahnya jarak tempuh.
- Kendaraan berbahan bakar **diesel** dan **gasoline** mendominasi data, namun cenderung memiliki pola distribusi harga yang berbeda.
- Beberapa kendaraan tetap memiliki harga tinggi meskipun mileage-nya besar, kemungkinan karena faktor lain seperti tahun produksi atau merek premium.

### EDA - Univariate Analysis

![price hist](https://github.com/user-attachments/assets/0e4715fd-3a50-4b48-b11d-1af6c936151f)

Gambar 10. Distribusi Harga Kendaraan

Gambar 10 memperlihatkan distribusi harga dari seluruh kendaraan dalam dataset. Mayoritas kendaraan memiliki harga di kisaran **$30.000 hingga $60.000**, dengan penurunan frekuensi pada harga-harga yang lebih tinggi. Kurva KDE (Kernel Density Estimate) menunjukkan bahwa distribusi cenderung miring ke kanan, mengindikasikan adanya sejumlah kendaraan dengan harga yang jauh lebih tinggi (outlier).

![10 vehicle](https://github.com/user-attachments/assets/704dbec7-1a64-42bc-9c1d-2511a253619a)

Gambar 11. 10 Merek Kendaraan Terbanyak

Gambar 11 menunjukkan sepuluh merek kendaraan yang paling banyak muncul dalam dataset. **Jeep, RAM, dan Dodge** mendominasi jumlah entri. Hal ini menunjukkan bahwa dataset memiliki dominasi merek tertentu yang kemungkinan berasal dari dealer atau sumber data spesifik. Informasi ini penting karena merek dapat menjadi salah satu faktor utama dalam model prediksi harga.

![fuel hist](https://github.com/user-attachments/assets/e02b6850-3353-4469-92ec-d1cf9a6c20bf)

Gambar 12. Distribusi Jenis Bahan Bakar

Gambar 12 menggambarkan jumlah kendaraan berdasarkan jenis bahan bakarnya. Terlihat bahwa **Gasoline** adalah bahan bakar yang paling umum, diikuti oleh **Diesel** dan jenis lainnya. Data ini penting untuk mengetahui preferensi pasar dan juga dapat digunakan sebagai variabel dalam prediksi harga, karena bahan bakar dapat memengaruhi nilai kendaraan.

![num hist](https://github.com/user-attachments/assets/a17f4482-d82a-43f2-9d52-2a3f9c22109f)

Gambar 13. Visualisasi semua kolom numerik

Gambar 13 memperlihatkan visualisasi semua kolom numerik dalam bentuk histogram. Hasil pengamatan menunjukkan
1. **Price**: Harga kendaraan menunjukkan distribusi miring ke kanan. Sebagian besar kendaraan berada di kisaran $30.000–$60.000, namun terdapat outlier dengan harga di atas $100.000.
2. **Mileage**: Sebagian besar kendaraan memiliki jarak tempuh rendah, antara 0–20 mil. Namun, terdapat beberapa kendaraan dengan mileage sangat tinggi yang menjadi outlier.
3. **Cylinders**: Nilai silinder paling umum adalah 4 dan 6, dengan sedikit kendaraan yang memiliki 8 silinder. Ini mencerminkan standar industri otomotif.
4. **Doors**: Sebagian besar kendaraan memiliki 4 pintu. Nilai lainnya seperti 2 atau 5 pintu jauh lebih sedikit.
5. **Year**: Mayoritas kendaraan adalah keluaran tahun 2024, menunjukkan bahwa dataset ini lebih fokus pada kendaraan baru.

![cat hist](https://github.com/user-attachments/assets/1ff50f3f-151a-44d5-ae80-f7da1501dffb)

Gambar 14. Visualisasi kolom kategorikal

Gambar 14 memperlihatkan visualisasi kolom kategorikal dalam bentuk histogram. Hasil pengamatan menunjukkan
1. **Make**: Merek kendaraan yang paling umum adalah Jeep, RAM, dan Dodge. Hal ini dapat menunjukkan dominasi merek tertentu dalam dataset.
2. **Model**: Beberapa model populer seperti Grand Cherokee dan 1500 muncul lebih sering. Namun, banyak model hanya muncul 1–2 kali.
3. **Fuel**: Jenis bahan bakar terbanyak adalah **Gasoline**, disusul oleh **Diesel**. Jenis bahan bakar lainnya sangat sedikit.
4. **Transmission**: Sebagian besar kendaraan menggunakan transmisi **Automatic** atau **8-Speed Automatic**.
5. **Trim**: Beberapa trim seperti Laramie, Big Horn, dan Denali cukup sering muncul. Trim mencerminkan varian fitur pada kendaraan.
6. **Body**: Tipe body **SUV** dan **Pickup Truck** mendominasi dataset. Jenis lain seperti sedan sangat sedikit.
7. **Drivetrain**: Jenis penggerak terbanyak adalah **Four-wheel Drive** dan **All-wheel Drive**, yang umum pada kendaraan SUV atau off-road.
8. **Exterior Color**: Warna seperti **White**, **Black**, dan **Silver** adalah warna eksterior yang paling sering digunakan.
9. **Interior Color**: **Black** dan **Global Black** mendominasi interior, mencerminkan tren desain interior yang populer.


## Data Preparation
Berikut merupakan data preparation yang diterapkan pada project ini :

1. Seleksi dan Pembersihan Fitur
- Fitur `description` dan `name` dihapus karena memiliki jumlah nilai unik yang sangat tinggi dan kurang relevan terhadap prediksi harga.
- Fitur `year` tetap dipertahankan karena berkaitan langsung dengan depresiasi harga kendaraan.

2. Encoding Fitur Kategorikal
Untuk mempersiapkan fitur kategorikal agar dapat digunakan dalam model regresi:
- Fitur dengan sedikit variasi nilai unik (misal `fuel`, `transmission`, `body`) di-encode menggunakan Label Encoding (konversi ke nilai integer).
- Fitur dengan nilai unik yang cukup banyak namun masih manageable (misal `make`, `exterior_color`) diolah dengan One-Hot Encoding jika diperlukan, namun dilakukan selektif untuk menghindari eksplosi dimensi.

3. Pemisahan Fitur dan Target
- Fitur target adalah `price`, yang dipisahkan ke dalam variabel y.
- Semua kolom fitur dimasukkan ke dalam variabel X, kecuali kolom `price`.

4. Train-Test Split
Data dibagi menjadi data latih dan data uji menggunakan rasio 80:20. Hal ini dilakukan untuk menghindari overfitting dan menguji performa model terhadap data yang belum pernah dilihat sebelumnya.

---

## Modeling

Tahap ini menjelaskan model machine learning regresi yang digunakan dalam proyek untuk memprediksi harga kendaraan. Masing-masing model dijelaskan dari sisi konsep, cara kerja pada dataset, serta alasan penggunaannya.

### 1. **Linear Regression**

**Linear Regression** digunakan sebagai model baseline karena kesederhanaannya dan kemampuannya menangkap hubungan linear antara fitur dan target.

#### Cara Kerja pada Dataset
Linear Regression mencoba mencari garis terbaik yang memetakan hubungan antara fitur-fitur kendaraan (seperti tahun, mileage, fuel, dll.) dengan harga (`price`). Model ini menghitung koefisien linear yang meminimalkan error antara prediksi dan nilai aktual.

#### Parameter yang Digunakan
- Model ini digunakan dengan parameter default (tanpa tuning).
- Tidak dilakukan regularisasi khusus karena fokus pada baseline.

### 2. **K-Nearest Neighbors Regressor (KNN)**

**KNN Regressor** memprediksi harga kendaraan berdasarkan rata-rata dari `k` tetangga terdekat dalam ruang fitur.

#### Cara Kerja pada Dataset
KNN menghitung jarak antara sebuah kendaraan baru dan kendaraan-kendaraan lain di dataset pelatihan, lalu mengambil rata-rata harga dari tetangga terdekat tersebut untuk menghasilkan prediksi.

#### Parameter yang Digunakan
- Model digunakan dengan nilai default untuk `k` (biasanya `k=5`).
- Jarak dihitung menggunakan metrik Euclidean.
- Tidak ada bobot khusus yang diterapkan pada tetangga.

### 3. **Decision Tree Regressor**

**Decision Tree** membagi dataset menjadi beberapa subset berdasarkan nilai fitur yang paling mempengaruhi harga, hingga diperoleh prediksi akhir.

#### Cara Kerja pada Dataset
Model ini membuat struktur pohon di mana setiap node merepresentasikan pengambilan keputusan berdasarkan fitur kendaraan. Hasil prediksi pada setiap leaf node adalah rata-rata dari data di subset tersebut.

#### Parameter yang Digunakan
- Model digunakan tanpa batasan khusus seperti `max_depth` atau `min_samples_split`.
- Menggunakan parameter default dari pustaka scikit-learn.

### 4. **Random Forest Regressor**

**Random Forest** adalah model ensemble yang menggabungkan banyak Decision Tree untuk menghasilkan prediksi yang lebih stabil dan akurat.

#### Cara Kerja pada Dataset
Random Forest membangun beberapa Decision Tree pada subset acak dari data dan fitur. Hasil prediksi dari semua pohon dirata-ratakan untuk menghasilkan prediksi akhir harga kendaraan.

#### Parameter yang Digunakan
- Model digunakan dengan parameter default (`n_estimators` default = 100).
- Tidak dilakukan tuning khusus dalam eksperimen ini.

---

## Evaluation

Keempat model di atas dievaluasi menggunakan metrik regresi: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, dan **R² Score**.  
Model **Random Forest** menunjukkan performa terbaik dalam memprediksi harga kendaraan dengan error paling rendah dan nilai R² tertinggi, sehingga dipilih sebagai model final.

Tabel 2. Evaluasi Model

Berikut adalah hasil evaluasi dari keempat model regresi yang digunakan:

| No | Model              | MAE        | RMSE       | R² Score |
|----|--------------------|------------|------------|----------|
| 1  | Random Forest       | 5695.85    | 10608.85   | 0.702    |
| 2  | Decision Tree       | 6226.55    | 11542.99   | 0.647    |
| 3  | Linear Regression   | 9316.23    | 14800.18   | 0.420    |
| 4  | K-Nearest Neighbors | 8408.07    | 14922.23   | 0.410    |

### Interpretasi:
- **Random Forest** memiliki **MAE dan RMSE paling rendah** serta **R² Score tertinggi**, menunjukkan model ini paling andal dalam memprediksi harga kendaraan berdasarkan fitur yang tersedia.
- **Decision Tree** juga memberikan hasil yang cukup baik, meskipun masih sedikit lebih tinggi error-nya dibanding Random Forest.
- **Linear Regression dan KNN** menunjukkan performa yang relatif lebih buruk karena kemungkinan tidak mampu menangkap hubungan non-linear dalam data.

Model dengan performa terbaik yaitu **Random Forest** direkomendasikan untuk diimplementasikan dalam sistem prediksi harga kendaraan pada tahap selanjutnya.

---

## Kesimpulan

Model **Random Forest** menunjukkan performa terbaik dengan nilai **R² Score sebesar 0.702** dan error yang paling rendah (**MAE = 5695.85**, **RMSE = 10608.85**), diikuti oleh **Decision Tree** dengan **R² Score sebesar 0.647**. Kedua model ini mampu menangkap hubungan kompleks antar fitur dalam data, terutama Random Forest yang menggunakan teknik ensemble untuk mengurangi overfitting.

Model **K-Nearest Neighbors (KNN)** menunjukkan kinerja yang cukup baik dengan **MAE sebesar 8408.07**, meskipun performanya sedikit lebih rendah dibandingkan model ensemble. Di sisi lain, model **Linear Regression** memiliki performa terendah dengan **R² Score hanya sebesar 0.420**, yang menunjukkan keterbatasannya dalam memodelkan relasi non-linear antar variabel dalam dataset ini.

Secara keseluruhan, **Random Forest** adalah pilihan terbaik untuk prediksi harga kendaraan bekas, karena mampu memberikan estimasi yang lebih akurat, stabil, dan dapat diandalkan dalam konteks pasar mobil yang sangat dinamis.

### Kesimpulan Dampak Model Terhadap Business Understanding

#### 1. Menjawab Problem Statements

- **Efektivitas Model Machine Learning**  
  Model Random Forest menunjukkan hasil prediksi yang sangat baik dalam memodelkan harga kendaraan bekas. Ini menunjukkan bahwa metode ensemble sangat cocok digunakan dalam domain pasar mobil yang kompleks.

- **Identifikasi Algoritma Terbaik**  
  Berdasarkan evaluasi metrik regresi, **Random Forest** merupakan algoritma terbaik, diikuti oleh **Decision Tree** dan **KNN**, sedangkan Linear Regression memberikan hasil yang kurang optimal.

#### 2. Mencapai Goals

- **Pembangunan Model Prediktif Akurat**  
  Dengan hasil evaluasi yang menunjukkan **R² Score di atas 0.70** untuk Random Forest, proyek ini telah berhasil membangun sistem prediksi harga kendaraan yang cukup presisi dan relevan untuk diterapkan dalam dunia nyata.

- **Perbandingan Kinerja Algoritma**  
  Evaluasi menyeluruh memberikan pemahaman yang jelas mengenai keunggulan dan keterbatasan tiap model, yang penting untuk pengambilan keputusan lebih lanjut di dunia bisnis otomotif.

#### 3. Solusi Statement

- **Eksplorasi dan Preprocessing Data**  
  Proses pembersihan data, penanganan missing value, encoding fitur kategorikal, serta pemisahan data latih dan uji membantu dalam membangun fondasi model yang kuat.

- **Penerapan Algoritma Machine Learning**  
  Penggunaan beberapa algoritma regresi memberi gambaran lengkap tentang mana model yang paling cocok. Random Forest menjadi unggulan karena performanya yang konsisten dan minim overfitting.

- **Penghapusan Duplikat dan Penanganan Nilai Hilang**  
  Dengan membersihkan data sebelum modeling, model yang dihasilkan menjadi lebih akurat dan tidak bias terhadap noise atau data yang tidak konsisten.


## Daftar Pustaka

1. A. Gupta, R. Kumar, dan S. Yadav, Predictive Modeling of Car Price using Random Forest, International Journal of Data Science and Analytics, vol. 10, no. 3, pp. 123-134, 2021. https://doi.org/10.1007/s41060-020-00230-5
2. N. Singh dan A. Kumari, "Comparison of Regression Algorithms in Car Price Prediction," Journal of Machine Learning and Applications, vol. 8, no. 2, pp. 45-56, 2020. https://doi.org/10.5120/jmla.2020.8202
