

# W-LSTMix: A Hybrid Modular Forecasting Framework for Trend and Pattern Learning in Short-Term Load Forecasting

**W-LSTMix** is a lightweight, modular hybrid forecasting model designed for building-level load forecasting across diverse building types. With approximately **0.13 million parameters**, W-LSTMix combines:

- **Wavelet-based signal decomposition**
- **N-BEATS** for ensemble forecasting
- **LSTM** for gated memory
- **MLP-Mixer** for efficient patch-wise mixing

This model achieves high forecasting accuracy with a minimal computational footprint.

---

## 📰 News

📢 **Our paper on W-LSTMix has been accepted at [ICML Workshop FMSD 2025](https://icml-structured-fm-workshop.github.io/)!**  
Check out the publication below:

> **W-LSTMix: A Hybrid Modular Forecasting Framework for Trend and Pattern Learning in Short-Term Load Forecasting**  
> *Shivam Dwivedi, Anuj Kumar, Harish Kumar Saravanan, Pandarasamy Arjunan*  
> In *Proceedings of the 1st ICML Workshop on Foundation Models for Structured Data, Vancouver, Canada. 2025*  
> [https://openreview.net/pdf?id=bG04Z3Jioc](https://openreview.net/pdf?id=bG04Z3Jioc)


---

## 🚀 Features

- Lightweight: ~0.13M parameters
- Modular design for flexible adaptation
- Effective generalization across building types
- Zero-shot capabilities
- Colab-ready demo

---

## 🛠 Installation

> ⚠️ It is recommended to use a separate virtual environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivDwd/W-LSTMix.git
   cd W-LSTMix
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the test dataset:**
   ```bash
   git clone https://huggingface.co/datasets/shivDwd/W_LSTMix_test_dataset
   ```

---

## 🧪 Running Tests

1. Change your working directory to the repo folder (if not already in it):
   ```bash
   cd W-LSTMix
   ```

2. Run the test script:
   ```bash
   python test.py
   ```

---

## 🗂 Notes

- ✅ Checkpoints for **zero-shot experiments** are provided in this repository.
- ⚙️ You can modify the configuration by editing the `config` file accordingly.

---

## 📓 Colab Quickstart

Use the following steps to try W-LSTMix on Google Colab:

```bash
!git clone https://github.com/shivDwd/W-LSTMix.git
%cd W-LSTMix
!git clone https://huggingface.co/datasets/shivDwd/W_LSTMix_test_dataset
!pip install -r requirements.txt
!python test.py
```

---




