# Laporan Proyek Machine Learning - M. Ardifa Rizqi Ramadhan
## Project Overview

platform pembelajaran daring seperti Coursera, edX, dan berbagai Massive Open Online Course (MOOC) lainnya telah mengalami pertumbuhan yang pesat dalam beberapa tahun terakhir. Ribuan kursus baru ditawarkan setiap tahunnya untuk memenuhi kebutuhan pendidikan yang semakin beragam dari masyarakat global. Namun, melimpahnya pilihan ini justru menimbulkan tantangan baru berupa information overload, di mana pengguna kesulitan memilih kursus yang paling sesuai dengan minat, kebutuhan, dan tingkat keahlian mereka. Apabila tidak diatasi, masalah ini dapat menyebabkan pengguna merasa frustasi, memilih kursus yang kurang relevan, hingga akhirnya menurunkan tingkat keterlibatan dan keberhasilan dalam pembelajaran.
Untuk membantu pengguna dalam menemukan kursus yang sesuai dengan minat, kebutuhan, dan tingkat keahlian mereka di tengah banyaknya pilihan, dibutuhkan sistem rekomendasi yang mampu menyajikan rekomendasi yang relevan. Proyek ini mengusulkan sistem rekomendasi berbasis konten (content-based filtering). Pendekatan berbasis konten ini dinilai efektif dalam situasi di mana data interaksi pengguna terbatas, seperti yang juga dibahas dalam penelitian tentang pengembangan sistem rekomendasi kursus berbasis konten [1].
Meskipun penelitian sebelumnya banyak menggunakan pendekatan berbasis kolaborasi, seperti yang dilakukan oleh Suryani et al. dalam mengoptimalkan item-based collaborative filtering untuk rekomendasi kursus di Coursera [2], pendekatan berbasis konten dipilih dalam proyek ini karena menyesuaikan dengan karakteristik dataset yang digunakan.Selain itu, untuk memastikan sistem rekomendasi ini benar-benar mampu memberikan rekomendasi yang relevan dan bermanfaat, perlu dilakukan evaluasi performa sistem menggunakan metrik yang sesuai. Sistem rekomendasi ini diharapkan tidak hanya meningkatkan relevansi rekomendasi bagi pengguna, tetapi juga memperkaya pengalaman mereka dalam menjelajahi pembelajaran daring.

## Business Understanding

Pengembangan sistem rekomendasi kursus pada platform Coursera memiki potensi untuk dapat memberikan manfaat bagi pengguna dan platform Coursera

### Problem Statements

- Bagaimana merancang sistem rekomendasi yang dapat membantu pengguna dalam memilih kursus yang relevan dari ribuan pilihan yang tersedia di platform Coursera?
- Bagaimana mengukur dan mengevaluasi performa sistem rekomendasi berbasis content-based filtering dalam menyajikan rekomendasi yang relevan?

### Goals

- Mengembangkan sistem rekomendasi berbasis content-based filtering yang mampu menyarankan kursus yang relevan bagi pengguna
- Mengevaluasi performa sistem rekomendasi yang dikembangkan menggunakan metrik yang sesuai

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari Kaggle, dengan tautan berikut: [Coursera-Courses-Dataset-2021.](https://www.kaggle.com/datasets/khusheekapoor/Coursera-courses-dataset-2021) Dataset ini berisi tentang kursus coursera dan fitur-fitur yang terkait, Berikut adalah gambaran singkat mengenai dataset tersebut:

### Variabel-variabel pada dataset Coursera Courses 2021 adalah sebagai berikut:
- Course Name: Nama Kursus
- University: Universitas penyelenggara kursus
- Dificult Level: Tingkat kesuliatan kursus(Beginner, Intermediate, Advance)
- Course Rating: Nilai kursus berdasarkan ulasan pengguna, biasanya dalam skala 1 hingga 5.
- Course URL: Tautan URL kursus di platform Coursera
- Course Desription: Deskripsi singkat mengenai isi dan tujuan kursus
- Skills: Ketrampilan atau kompetensi yang akan dipelajari dalam kursus

### Kondisi Data
- Jumlah baris dan Kolom: dataset ini berisi 3522 baris dan 7 kolom
- seluruh data bertipe data object(string atau teks)
- Tidak terdapat data kosong (null) di setiap kolom

### Exploratory Data Analysis (EDA)
#### Struktur Data

 Tabel 1. Deskripsi dan tipe data variabel  
| #   | Column    | Non-Null Count | Dtype   |
|-----|-----------|----------------|---------|
| 0   | Course Name| 3522 non-null  | object  |
| 1   | University | 3522 non-null  | object  |
| 2   | Dificulut Level| 3522 non-null  | object   |
| 3   | Course Rating| 3522 non-null  | object   |
| 4   | Course URL| 3522 non-null  | object  |
| 5   | Course Description| 3522 non-null| object|
| 6   | Skills    | 3522 non-null  | object |

Dataset ini berisi 3522 baris dan 7 kolom, seluruh data bertipe data object(string atau teks), Tidak terdapat data kosong (null) di setiap kolom

#### Missing Value

Tabel 2. Missing Value
|  Column  |  0  | 
|----------|-----|
| Course Name|  0  |
| University |  0  |
| Dificult Level|  0  |
| Course Rating|  0  |
| Course URL|  0  |
| Course Description|  0  |
| Skills|  0  |

Tidak terdapat missing values pada keenam kolom (Course Name, University, Dificult Level, Course Rating, Course URL, Course Description) dalam dataset.

#### Bar Chart fitur Dificult Level

![Image](https://github.com/user-attachments/assets/478c5437-d76b-44a4-9cd6-a8782f6fb402)

Gambar 1. Bar Chart Variabel Dificult Level

Berdasarkan Visualisasi diatas menunjukkan bahwa:
Mayoritas kursus ditujukan untuk pemula (Beginner) dengan 41% (1444 kursus), menunjukkan target utamanya adalah pengguna yang baru memulai pembelajaran di bidang tertentu. Ini bisa menjadi pertimbangan penting bagi lembaga pendidikan yang ingin menyasar segmen pemula.

## Data Preparation

### Tahapan Data Preparation
#### Handling Missing Value
- Dilakukan pengecekan terhadap keberadaan missing value pada dataset menggunakan fungsi isnull().sum().Hasil dari pengecekan ini menunjukkan bahwa seluruh kolom dalam dataset tidak memiliki nilai yang hilang. Dengan demikian, tidak diperlukan penanganan lanjutan untuk mengisi atau menghapus data yang kosong.
- Proses ini tetap penting untuk memastikan kualitas data sebelum masuk ke tahap pemodelan.
#### Feature Selection
- Memilih kolom-kolom yang relevan untuk digunakan dalam sistem rekomendasi berbasis content-based filtering. Dalam hal ini, kolom yang dipilih adalah Course Name, Course Description, dan Skills. Ketiga kolom ini dianggap paling menggambarkan isi atau konten dari masing-masing kursus. Course Name digunakan sebagai identitas kursus, sedangkan Course Description dan Skills akan menjadi dasar dalam menghitung kemiripan antar kursus.
- Pemilihan fitur yang tepat sangat penting agar sistem dapat merekomendasikan kursus lain yang benar-benar serupa dari segi konten dan keterampilan yang ditawarkan.
#### Column Renaming
- Nama kolom yang mengandung spasi seperti Course Name dan Course Description diubah menjadi Course_Name dan Course_Description.
- Penggantian nama ini bertujuan untuk meningkatkan konsistensi dalam penamaan dan mempermudah akses kolom selama proses pengolahan data selanjutnya. Nama kolom yang tidak mengandung spasi akan lebih mudah digunakan dalam penulisan kode Python dan menghindari potensi error saat pemanggilan nama kolom. Dengan demikian, proses ini merupakan bagian penting dari penataan struktur data sebelum masuk ke tahap pembuatan model rekomendasi.
#### TF-IDF Vectorizer
- Data teks di kolom Course_Name diproses menggunakan TF-IDF dengan TfidfVectorizer, dimulai dari perhitungan IDF, transformasi ke matriks TF-IDF, hingga dikonversi ke dense matrix dan ditampilkan dalam bentuk dataframe untuk pengecekan.
- Teknik TF-IDF digunakan agar data teks memiliki bobot numerik yang mencerminkan pentingnya setiap kata, sehingga algoritma machine learning dapat fokus pada kata-kata yang lebih relevan dan tidak terpengaruh oleh kata yang terlalu umum.
  
## Modeling

Pada tahap model development dengan Content Based Filtering menggunakan tahapan berikut:
### Cosine Similarity
Setelah mendapatkan representasi vektor dari setiap kursus, tahap selanjutnya adalah menghitung derajat kemiripan antar kursus menggunakan teknik Cosine Similarity. Metode ini digunakan karena mampu mengukur kemiripan dua vektor dalam ruang berdimensi tinggi tanpa terpengaruh oleh panjang dokumen. Hasil perhitungan cosine similarity berupa nilai antara 0 hingga 1, di mana nilai mendekati 1 menandakan bahwa dua kursus sangat mirip. Matriks hasil perhitungan tersebut kemudian disusun dalam bentuk DataFrame dua dimensi untuk memudahkan pencarian kursus yang memiliki nilai kemiripan tertinggi terhadap kursus tertentu.
### Top-N Recommendation
Dari hasil perhitungan cosine similarity, didapatkan sejumlah kandidat kursus yang memiliki kemiripan tinggi dengan kursus yang dipilih pengguna. Selanjutnya, kandidat tersebut disusun berdasarkan urutan nilai similarity tertinggi, dan sejumlah Top-N kursus ditampilkan sebagai hasil rekomendasi. Dalam proyek ini, ditentukan bahwa sistem akan menampilkan 5 rekomendasi teratas (Top-5 recommendation). Setiap rekomendasi disajikan bersama deskripsi dan keterampilan yang relevan, sehingga pengguna dapat mempertimbangkan kursus lain yang memiliki karakteristik serupa dengan yang telah dipilih sebelumnya. Hasil dari penerapannya dengan judul 'Predictive Analytics and Data Mining' dapat dilihat pada tabel berikut:

Tabel 3. Top-5 Recommended
| Course_Name                                     | Course_Description                                                                                           | Skills                                                     |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| Predictive Modeling and Analytics               | Welcome to the second course in the Data Analytics for Business specialization.                              | Logistic Regression, analytics, predictive analytics       |
| Data Mining Project                             | Note: You should complete all the other courses before beginning this project-based course.                  | ordered pair, tableau software, Similarity Measures        |
| Design Thinking and Predictive Analytics for... | This is the second course in the four-course specialization ‘Data Science and Agile Systems for Product M... | Logistic Regression, supervised learning, gradient descent |
| Text Mining and Analytics                       | This course will cover the major techniques for mining and analyzing text data.                              | probability, Topic Model, Natural Language Processing      |
| Population Health: Predictive Analytics         | Predictive analytics has a longstanding tradition in the area of healthcare and public health.               | sample size determination, prognostics, external validation|


## Evaluation

Pada proyek ini, sistem rekomendasi dikembangkan dengan pendekatan content-based filtering, yang memanfaatkan informasi variabel course seperti Nama Course, Course Deskription, dan skills yang ditawarkan. Untuk mengevaluasi performa sistem rekomendasi, digunakan metrik Precision yang dipilih karena sesuai dengan tujuan proyek, yaitu memberikan rekomendasi course yang relevan berdasarkan kemiripan konten.

Untuk mengukur performa sistem, dilakukan pengujian dengan input course berjudul “Predictive Analytics and Data Mining”. Sistem kemudian menghasilkan lima rekomendasi teratas, yaitu: Predictive Modeling and Analytics, Data Mining Project, Design Thinking and Predictive Analytics for Data Products, Text Mining and Analytics, dan Population Health: Predictive Analytics. Setelah dilakukan evaluasi terhadap masing-masing course yang direkomendasikan, ditemukan bahwa seluruh rekomendasi memiliki relevansi yang kuat dengan course input, baik dari sisi topik utama, seperti predictive analytics dan data mining, maupun dari keterampilan yang ditawarkan seperti logistic regression, supervised learning, dan natural language processing. Berdasarkan hal tersebut, diperoleh nilai Precision sebesar 5/5 atau 100%, yang berarti seluruh rekomendasi yang diberikan sistem dinilai relevan.

## References:
[1] A. A. Neamah and A. S. El-Ameer, "Design and Evaluation of a Course Recommender System Using Content-Based Approach," in Proceedings of the 2018 1st International Conference on Advanced Science and Engineering (ICOASE), doi: 10.1109/ICOASE.2018.8548789 (https://doi.org/10.1109/ICOASE.2018.8548789).

[2]  M. Suryani, R. Pangestu, and A. Pradana, “Optimizing Item-based Collaborative Filtering for Course Recommendation: a Case Study with Coursera Data,” in Proceedings of the 2024 International Conference on Information Technology Systems and Innovation (ICITSI), doi: 10.1109/ICITSI65188.2024.10929184 (https://doi.org/10.1109/ICITSI65188.2024.10929184)

**---Ini adalah bagian akhir laporan---**
