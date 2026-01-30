# eye-disease-decision-support-system
---

# 🧠 Göz Hastalıkları Karar Destek Sistemi

*(Eye Disease Decision Support System)*

Bu proje, **fundus kamera görüntüleri** kullanılarak göz hastalıklarının otomatik olarak tespit edilmesini amaçlayan **CNN tabanlı bir Karar Destek Sistemi**dir.
Sistem, **Derin Öğrenme (ResNet50)** mimarisi ile geliştirilmiş ve **Flask web arayüzü** üzerinden kullanıcıların görüntü yükleyerek tahmin almasına olanak sağlamaktadır.

---

## 📌 Proje Özellikleri

* 10 sınıflı göz hastalığı sınıflandırması
* **Transfer Learning (ResNet50)** tabanlı CNN modeli
* Flask tabanlı web arayüz
* Yüklenen görüntünün:

  * Gösterilmesi
  * Hastalık tahmini
  * Güven oranı (%) ile sunulması
* Basit ve kullanıcı dostu arayüz

---

## 🦠 Tespit Edilen Göz Hastalıkları

Sistem aşağıdaki sınıfları tanımaktadır:

1. Central Serous Chorioretinopathy
2. Diabetic Retinopathy
3. Disc Edema
4. Glaucoma
5. Healthy
6. Macular Scar
7. Myopia
8. Pterygium
9. Retinal Detachment
10. Retinitis Pigmentosa

---

## 🗂️ Veri Seti

* **Kaynak:** Mendeley Data – Fundus Göz Görüntüleri
* Veri seti, fundus kamera ile elde edilmiş retinal görüntülerden oluşmaktadır.
* Görüntüler sınıf bazlı klasörler halinde düzenlenmiştir.

---

## ⚙️ Kullanılan Teknolojiler

* **Python 3.11**
* **TensorFlow / Keras**
* **NumPy**
* **OpenCV**
* **Matplotlib**
* **Flask**
* **HTML & CSS**

---

## 🏗️ Proje Yapısı

```
eye_project/
│
├── app.py                 # Flask uygulaması
├── train.py               # Model eğitim kodu
├── model/
│   └── eye_disease_model.h5
├── dataset/               # Göz görüntüleri veri seti
├── templates/
│   └── index.html         # Web arayüzü
├── static/                # Yüklenen görüntüler
├── accuracy.png           # Eğitim doğruluk grafiği
├── loss.png               # Eğitim kayıp grafiği
├── venv/                  # Sanal ortam
└── README.md
```

---

## 🚀 Projenin Çalıştırılması

### 1️⃣ Sanal Ortamı Aktifleştirme

```bash
venv\Scripts\activate
```

### 2️⃣ Flask Uygulamasını Çalıştırma

```bash
python app.py
```

### 3️⃣ Web Arayüzüne Erişim

Tarayıcıdan aşağıdaki adrese gidiniz:

```
http://127.0.0.1:5000/
```

### 4️⃣ Görüntü Yükleme ve Tahmin

* Fundus görüntüsünü yükleyin
* **Tahmin** butonuna basın
* Hastalık adı ve güven oranını görüntüleyin

---

## 🧪 Model Eğitimi

* **Temel Model:** ResNet50 (ImageNet ön-eğitimli)
* **Kayıp Fonksiyonu:** `categorical_crossentropy`
* **Optimizasyon:** Adam
* **Epoch Sayısı:** 10
* Eğitim CPU ortamında gerçekleştirilmiştir.

Eğitim sürecine ait grafikler:

* `accuracy.png`
* `loss.png`

---

## 📊 Sonuçlar

Geliştirilen sistem, retinal fundus görüntülerinden göz hastalıklarını başarıyla sınıflandırabilmektedir.
Çok sınıflı (10 sınıf) yapı ve veri setinin karmaşıklığı nedeniyle güven oranları orta seviyede olsa da, sistem derin öğrenme tabanlı karar destek yaklaşımlarının göz hastalıklarının tespitinde uygulanabilirliğini göstermektedir.
