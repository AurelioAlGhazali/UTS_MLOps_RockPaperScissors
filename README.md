# Rock-Paper-Scissors Classifier – MLOps UTS

Proyek ini merupakan bagian dari UTS mata kuliah MLOps. Model yang dibangun melakukan klasifikasi gesture tangan (rock, paper, scissors).

## 1. Tech Stack

- Python
- PyTorch (ResNet18, transfer learning)
- Streamlit (deployment)
- GitHub + Streamlit Cloud

## 2. Dataset

- Dataset: real-world rock–paper–scissors hand gesture images
- Sumber: https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset
- Split:
  - Train: 1020 gambar
  - Validation: 804 gambar
  - Test: 540 gambar (local evaluation)

**Catatan:** Dataset tidak di-commit ke repository karena ukuran & lisensi. Dapat diunduh dari link di atas.

## 3. Training

Script/notebook utama:

- 'UTS_MLOps_RockPaperScissors.ipynb'

Langkah umum:
1. Load dataset (train/val/test)
2. Augmentasi (RandomResizedCrop, Flip, dsb)
3. Fine-tune ResNet18
4. Simpan checkpoint ke: `checkpoints_rps_real/resnet18_rps_real_best.pth`

## 4. Deployment (Streamlit)

File utama: `app.py`

Cara run lokal:

```bash
pip install -r requirements.txt
streamlit run app.py


