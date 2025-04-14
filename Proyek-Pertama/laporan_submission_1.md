# Laporan Proyek Machine Learning - M. Ardifa Rizqi Ramadhan
## Domain Proyek
Penentuan harga yang akurat merupakan faktor krusial yang mempengaruhi profitabilitas dan daya saing perusahaan dalam industri jual-beli mobil bekas. Harga mobil bekas tidak memiliki standar baku seperti komoditas lainnya, sehingga dipengaruhi oleh berbagai fitur seperti merk, model, tahun produksi, jarak tempuh, kondisi kendaraan dan fitur tambahan. Variabilitas ini menuntut perusahaan untuk memiliki sistem yang mampu memprediksi harga pasar secara tepat guna mendukung pengambilan keputusan bisnis yang strategis.
Sebagai perusahaan yang bergerak di bidang distributor dan ritel mobil bekas, penetapan harga beli yang tepat sangat penting untuk memastikan margin keuntungan tetap optimal. Oleh karena itu, perusahaan membutuhkan sistem prediksi harga yang akurat untuk mendukung pengambilan keputusan dalam proses pembelian dan penjualan kendaraan. Masalah ini perlu diselesaikan karena keputusan harga yang tidak tepat dapat merugikan bisnis, dan sistem prediksi otomatis berbasis data menjadi solusi untuk meningkatkan akurasi dan efisiensi penentuan harga [1]. 

Referensi:
[1] Balcıoğlu, M. B., & Sezen, Y. (2023). Car Price Prediction Using Machine Learning Techniques. International Journal of Artificial Intelligence and Data Mining, 11(1), 25-34.
[2] Ezenkwu, C. P., Daramola, O., & Adigun, A. A. (2021). Comparative Analysis of Ensemble Learning Techniques for Car Price Prediction. Journal of Engineering and Applied Sciences, 16(9), 2213–2221.

## Business Understanding

### Problem Statements

- Fitur apa yang paling berpengaruh terhadap harga mobil bekas?
- Bagaimana model Machine Learning mampu memprediksi harga mobil bekas berdasarkan fitur-fitur yang tersedia?
- Algoritma Machine Learning Regresi apa yang memberikan hasil prediksi terbaik berdasarkan metrik evaluasi seperti MAE, MSE, atau RMSE?

### Goals

- Mengidentifikasi fitur-fitur yang paling berkorelasi dengan harga mobil bekas.
- Membangun model machine learning untuk memprediksi harga mobil bekas berdasarkan fitur-fitur yang tersedia.
- Mengevaluasi dan menentukan algoritma regresi terbaik berdasarkan performa metrik regresi (MAE, MSE, dan RMSE).

    ### Solution statements
    - Melakukan Exploratory Data Analysis (EDA) untuk memahami pola dan hubungan antar fitur, serta mengidentifikasi fitur yang paling memengaruhi harga mobil menggunakan analisis korelasi dan visualisasi.
    - Membangun dan membandingkan tiga algoritma regresi: Linear Regression, Random Forest Regressor, dan Support Vector Regressor (SVR).
    - Mengevaluasi performa model menggunakan metrik evaluasi regresi seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), dan Root Mean Squared Error (RMSE).

## Data Understanding

Dataset yang digunakan dalam proyek ini diambil dari Kaggle, dengan tautan berikut: [Car-Prices-Prediction-data](https://www.kaggle.com/datasets/mrsimple07/car-prices-prediction-data/data). Dataset ini berisi tentang harga mobil dan fitur-fitur yang terkait, Berikut adalah gambaran singkat mengenai dataset tersebut:

### Variabel-variabel pada Car-Price-Prediction-Data adalah sebagai berikut:
- Make : Merek atau produsen mobil. (misalnya, Toyota, Honda, Ford)
- Model : Model mobil tertentu. (misalnya, Camry, Civic, F-150)
- Year : Tahun produksi mobil.
- Mileage : Total jarak tempuh mobil. (dalam mil)
- Condition : Kondisi mobil dikategorikan sebagai Excellent, Good, atau Fair.
- Price : Harga mobil

### Kondisi Data
- Tipe Data:
Himpunan data terdiri dari tipe data numerik dan kategoris. Fitur numerik termasuk Year, Mileage, dan Price, sedangkan Kategoris termasuk Make, Model, dan Condition
- Jumlah Baris dan Kolom:
Dataset memiliki 1000 baris dan 7 kolom 

### Exploratory Data Analysis (EDA)
#### Struktur Data

 Tabel 1. Deskripsi dan tipe data variabel  
| #   | Column    | Non-Null Count | Dtype   |
|-----|-----------|----------------|---------|
| 0   | Make      | 1000 non-null  | object  |
| 1   | Model     | 1000 non-null  | object  |
| 2   | Year      | 1000 non-null  | int64   |
| 3   | Mileage   | 1000 non-null  | int64   |
| 4   | Condition | 1000 non-null  | object  |
| 5   | Price     | 1000 non-null  | float64 |

Setelah Menghapus Kolom Unnamed: 0 yang hanya berisi index dari file CSV, Dataset kini memiliki 1000 baris dan 6 kolom. Kolom Make, Model, Year, Mileage, dan Condition merupakan Fitur dan kolom Price merupakan target/label

#### Deskriptif Statistik

Tabel 2.  Deskriptif Statistik fitur numerik
|  #  |Year 	 |Mileage      |Price       |
|-----|----------|-------------|------------|
|count|1000.00000|1000.000000  |1000.000000 |
|mean |2015.86500|78796.927000 |22195.205650|
|std  |3.78247	 |39842.259941 |4245.191585 |
|min  |2010.00000|10079.000000 |12613.000000|
|25%  |2013.00000|44942.750000 |18961.862500|
|50%  |2016.00000|78056.500000 |22247.875000|
|75%  |2019.00000|112366.250000|25510.275000|
|max  |2022.00000|149794.000000|31414.900000|

Berdasarkan informasi diatas menunjukkan bahwa:
- Tahun mobil berkisar antara 2010 hingga 2022, dengan median di 2016. Ini menunjukkan mayoritas mobil relatif baru.
- Kisaran jarak tempuh sangat bervariasi (10 ribu hingga hampir 150 ribu km), menunjukkan dataset mencakup mobil baru hingga cukup tua.
- Harga mobil tersebar antara sekitar 12 ribu hingga 31 ribu, dengan rata-rata sekitar 22 ribu. Ini cocok untuk kasus regresi harga mobil.
  
#### Missing Value

Tabel 3. Missing Value
|  Column  |  0  | 
|----------|-----|
| Make     |  0  |
| Model    |  0  |
| Year     |  0  |
| Mileage  |  0  |
| Condition|  0  |
| Price    |  0  |
| Make     |  0  |

Tidak terdapat missing values pada keenam kolom (Make, Model, Year, Mileage, Condition, dan Price) dalam dataset.

#### Univariate Analysis

##### Fitur Make (Merek Mobil)

link

Gambar 1. Bar Chart fitur Make

Berdasarkan Visualisasi diatas menunjukkan bahwa:
- Chevrolet adalah merek paling umum dalam dataset, disusul oleh Toyota.
- Ford dan Honda memiliki jumlah yang sama (199 sampel).
- Nissan merupakan merek dengan frekuensi paling rendah, namun perbedaannya sangat tipis.
- Distribusi merek cukup merata, masing-masing menyumbang sekitar 18–21% dari total data.

##### Fitur Condition (Kondisi Mobil)

link

Gambar 2. Bar Chart fitur Condition

Berdasarkan visualisasi diatas menunjukkan bahwa:
- Mayoritas mobil (59.5%) berada dalam kondisi Excellent, menunjukkan bahwa sebagian besar kendaraan dalam dataset ini berada dalam kondisi sangat baik.
- Hanya 11.3% mobil yang berada dalam kondisi Fair

##### Histogram Fitur Numerik (Year, Mileage, Price)

link

Gambar 3. Histogram Fitur Numerik

Berdasarkan Visualisasi Histogram diatas menunjukkan bahwa:
- Pada Fitur Year yang merupakan tahun produksi mobil, menunjukkan distribusi terlihat cukup merata pada tahun 2010 hingga 2022, dengan puncak jumlah kendaraan berapa di tahun 2011 dan 2015.
- Pada fitur Mileage yang merupakan jarak tempuh mobil, menunjukan bahwa distribusi terlihat cukup menyebar, mulai dari sekitar 10.000 km hingga lebih dari 140.000 km
- Pada fitur Prices yang merupakan harga mobil, menunjukkan bahwa sebagian besar mobil berada pada kisaran harga pasar menengah

#### Multivariate Analysis

##### Heatmap Korelasi Fitur Numerik

link

Gambar 4. Heatmap Korelasi Fitur Numerik

Berdasarkan visualisasi heatmap korelasi diatas menunjukkan bahwa:
- Fitur Year memiliki pengaruh paling besar terhadap Price, dengan korelasi negatif yang sangat kuat.
- Mileage juga berpengaruh terhadap Price, meskipun tidak sekuat Year.

## Data Preparation

### Tahapan Data Preparation
#### One Hot Encoding
- Mengubah fitur kategorikal (berupa teks) menjadi fitur baru yang sesuai sehingga dapat mewakili variabel kategori
- Proses ini menghasilkan kolom baru sebanyak jumlah kategori unik di tiap fitur.
- proses ini harus dilakukan karena Algoritma machine learning (seperti regresi dan random forest) hanya bekerja dengan data numerik.
#### Feature target Split 
- Memisahkan data menjadi dua bagian: fitur (X) dan target prediksi (y).
- Kolom Price digunakan sebagai target (y), dan semua kolom lain digunakan sebagai fitur(X).
- Proses ini harus dilakukan karena model harus jelas mana input(fitur) dan output(target).
#### Train-Test Split
- Membagi data menjadi dua subset: data pelatihan (training) dan data.pengujian (testing), agar bisa mengukur performa model secara objektif
- Data dibagi dengan rasio 80% training dan 20% testing.
- Proses ini perlu dilakukan karena untuk memastikan bahwa model benar-benar berguna, bukan sekedar jago di data latih.

## Modeling
Pada tahap Modeling, akan menggunakan tiga Algoritma Machine Learning untuk tugas Regresi, yaitu: Linear Regression, Random Forest Regressor, dan Suport Vector Regressor. Kinerja ketiga algoritma akan dievaluasi menggunakan metrik R², RMSE, dan MAE, kemudian dibandingkan untuk memilih satu model terbaik.

### Linear Regression
model statistik paling dasar yang memodelkan hubungan linear antara variabel independen (fitur) dan dependen (target). Model ini mencari garis lurus terbaik yang meminimalkan jumlah kesalahan kuadrat (ordinary least squares). Keunggulan utamanya terletak pada kecepatan komputasi dan kemudahan interpretasi koefisien, membuatnya ideal sebagai baseline model. Namun, model ini memiliki keterbatasan dalam menangkap hubungan non-linear dan sangat sensitif terhadap outlier serta multikolinearitas antar fitur.
Parameter yang digunakan:
- Default scikit-learn
- Tidak ada hyperparameter tuning karena model sederhana

### Random Forest Regressor 
metode ensemble yang membangun banyak decision tree secara paralel dan menggabungkan prediksinya. Setiap tree dilatih pada subset data dan fitur yang berbeda (teknik bagging) untuk meningkatkan akurasi dan stabilitas prediksi. Kelebihan utama algoritma ini adalah kemampuannya menangkap pola kompleks dan non-linear, serta ketahanannya terhadap overfitting berkat mekanisme averaging. Meskipun lebih lambat dan kurang interpretabel dibanding linear regression, Random Forest tidak memerlukan preprocessing ekstensif seperti scaling dan dapat menangani missing value secara implisit.
Parameter yang digunakan:
- random_state=42

### Support Vector Regression (SVR)
Support Vector Regression (SVR) mengadaptasi konsep Support Vector Machine untuk masalah regresi dengan mencari hyperplane optimal yang mempertahankan kesalahan prediksi dalam batas toleransi ε. Menggunakan kernel trick (seperti RBF), SVR mampu memodelkan hubungan non-linear dengan memproyeksikan data ke dimensi lebih tinggi. Keunggulannya terletak pada efektivitas di data berdimensi tinggi dan ketahanan terhadap overfitting, namun memerlukan tuning hyperparameter yang hati-hati dan preprocessing berupa feature scaling. Kompleksitas komputasinya yang tinggi membuat SVR kurang cocok untuk dataset berskala besar.
Parameter yang digunakan:
- Default Scikit-learn

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.
Pada tahap evaluasi, ketiga model Machine Learning dalam memprediksi harga mobil menggunakan metrik yang relevan untuk masalah regresi. Metrik yang digunakan adalam R-Square(R²), RMSE (Root Mean Squared Error), dan MAE (Mean Absolute Error).

### Metrik Evaluasi yang digunakan
- R-Square(R²): Mengukur seberapa baik model menjelaskan variasi data target.
- Mean Squared Error (MSE): Memberikan penalti lebih besar untuk kesalahan yang besar.
- Mean Absolute Error (MAE): Mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi.

### Hasil evaluasi

|      Model      |R² Score|     RMSE  |     MAE   |
|-----------------|--------|-----------|-----------|
|Linear Regression|1.000000|  0.069118 |   0.060029|
|Random Forest    |0.999071| 137.197834|  97.705023|
|SVR              |0.009285|4480.515800|3718.962264|

Insight:
- Berdasarkan evaluasi menggunakan metrik R², RMSE, dan MAE, model Linear Regression memiliki performa terbaik dalam memprediksi harga mobil.
- Random Forest layak dipertimbangkan sebagai solusi jika error RMSE/MAE masuk akal dalam konteks bisnis.
- Support Vector Regression perlu preprocessing dan tuning untuk bisa kompetitif.

**---Ini adalah bagian akhir laporan---**
