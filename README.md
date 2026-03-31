# 🖼️ Text-to-Image Synthesis using CLIP & BigGAN

A Vision-Language Model (VLM) project that uses **OpenAI CLIP** to perform semantic text-to-image retrieval — finding the image from a candidate set that best matches a given natural language description.

---

## 📌 Project Overview

This project demonstrates how large pre-trained Vision-Language Models can understand both text and images in a shared embedding space. Given a text prompt (e.g., `"A happy dog"`), the system ranks a set of candidate images by semantic similarity and returns the best match.

**Key concepts applied:**
- Multi-modal embeddings (text + image in shared space)
- Zero-shot image retrieval using cosine similarity
- CLIP (Contrastive Language–Image Pretraining) from OpenAI
- BigGAN for understanding generative synthesis pipelines

---

## 🗂️ Repository Structure

```
text-to-image-clip/
├── Text_to_Image_Synthesis_LLMs_VLMs.ipynb   # Main Colab notebook
├── README.md                                  # This file
└── requirements.txt                           # Python dependencies
```

---

## ⚙️ Setup Instructions

### Option 1: Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Text_to_Image_Synthesis_LLMs_VLMs.ipynb` or open it directly from GitHub
3. Set Runtime → **GPU** (T4 recommended)
4. Run all cells in order

### Option 2: Run Locally

**Prerequisites:** Python 3.8+, pip, CUDA (optional but recommended)

```bash
# Clone the repository
git clone https://github.com/<your-username>/text-to-image-clip.git
cd text-to-image-clip

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Launch Jupyter
jupyter notebook Text_to_Image_Synthesis_LLMs_VLMs.ipynb
```

---

## 📦 Dependencies

```
torch
torchvision
ftfy
regex
tqdm
Pillow
requests
clip @ git+https://github.com/openai/CLIP.git
```

Install all at once:
```bash
pip install ftfy regex tqdm Pillow requests
pip install git+https://github.com/openai/CLIP.git
```

---

## 🚀 How to Use

### Step 1 – Load the CLIP model
The notebook automatically downloads the `ViT-B/32` variant of CLIP and loads it onto GPU (or CPU if unavailable).

### Step 2 – Provide candidate image URLs
Edit the `urls` list in the notebook to point to any publicly accessible images:

```python
urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    # add more...
]
```

### Step 3 – Set your text prompt
Change the `text` variable to whatever you want to search for:

```python
text = "A happy dog"      # try: "A snowy mountain", "A red sports car", etc.
```

### Step 4 – Run all cells
The notebook will:
1. Download and preprocess all candidate images
2. Encode both the text and images using CLIP
3. Compute cosine similarity scores
4. Display the best-matching image

---

## 🧠 How It Works

```
Text Prompt  ──► CLIP Text Encoder  ──►  Text Embedding  ──┐
                                                             ├──► Cosine Similarity ──► Best Match
Images       ──► CLIP Image Encoder ──►  Image Embeddings ──┘
```

CLIP is trained on 400M (image, text) pairs using contrastive learning. Both modalities are projected into a shared 512-dimensional embedding space. The image whose embedding is closest (highest dot product) to the text embedding is returned as the best match.

---

## 📊 Example Results

| Text Prompt | Top Match Description |
|---|---|
| `"A happy dog"` | Smiling golden retriever photo |
| `"A waterfall in nature"` | Plitvice Lakes, Croatia |
| `"A futuristic city"` | Sci-fi skyline illustration |

---

## 📚 References

- [CLIP Paper – Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [Vision Language Models – HuggingFace Blog](https://huggingface.co/blog/vlms)
- [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- [BigGAN – Brock et al., 2018](https://arxiv.org/abs/1809.11096)

---

## 👥 Authors

Developed as part of the **EPICS Phase II** project at **VIT Bhopal University**.

---

## 📄 License

This project is intended for academic and educational use only.
