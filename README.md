# 🚀 Master LLM Chat - Custom Transformer Language Model

A complete implementation of a custom transformer-based language model with an interactive chat interface built using Gradio. This project features a modern, responsive UI with neon-green theme and advanced neural architecture components.

## ✨ Features

### **Neural Architecture**
- **Custom Transformer Implementation**: Built from scratch with modular components
- **Multi-Head Attention**: Efficient attention mechanisms for better context understanding
- **Rotary Position Encoding**: Advanced positional encoding for improved sequence modeling
- **Custom BPE Tokenizer**: Optimized tokenization with 10,500+ vocabulary size
- **Layer Normalization**: Custom implementation for training stability
- **GELU Activation**: Modern activation function in MLP layers

### **Modern UI/UX**
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Neon Green Theme**: Eye-catching dark theme with neon accents
- **Glassmorphism Effects**: Modern blur and transparency effects
- **Floating Action Buttons**: Intuitive interaction controls
- **Real-time Chat**: Instant model responses with typing indicators
- **Theme Toggle**: Dynamic light/dark mode switching

### **Advanced Controls**
- **Temperature Control**: Adjust response creativity (0.1-2.0)
- **Top-k Sampling**: Control vocabulary diversity (1-100)
- **Top-p (Nucleus) Sampling**: Fine-tune probability distribution (0.1-1.0)
- **Token Limit**: Configurable response length (1-30 tokens)
- **Quick Presets**: One-click settings for different conversation styles

### 🔧 **Technical Features**
- **CPU Optimized**: Efficient inference without GPU requirements
- **Model Management**: Load custom models from URLs or local files
- **Advanced Sampling**: Multiple sampling strategies for diverse responses
- **Error Handling**: Robust error management and user feedback
- **Extensible Architecture**: Modular design for easy customization

## Project Structure

```
llm-from-master/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
│
├── training/                       # Training resources
│   └── master_llm_trainer.ipynb   # Model training notebook
│
└── version_v1/                     # Model architecture
    ├── master_model.py             # Main transformer model
    ├── master_tokenizer.py         # BPE tokenizer implementation
    ├── master_embedding.py         # Token + positional embeddings
    ├── master_decoder_block.py     # Transformer decoder block
    ├── master_multi_head_attention.py  # Attention mechanism
    ├── master_layer_norm.py        # Layer normalization
    ├── master_mlp.py               # Feed-forward network
    ├── master_causal_attention.py  # Causal attention implementation
    ├── my_bpe_tokenizer.json       # Trained tokenizer file
    └── u_model.pth                 # Pre-trained model weights
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-from-master.git
   cd llm-from-master
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to the provided local URL (usually `http://127.0.0.1:7860`)

##  Usage

### Basic Chat
1. Type your message in the input field
2. Press Enter or click the "Send" button
3. Adjust generation parameters using the settings panel
4. Use floating action buttons for quick actions

### Generation Controls
- **Temperature**: Lower values (0.1-0.5) for focused responses, higher (1.0-2.0) for creativity
- **Top-k**: Limit vocabulary to top k tokens (recommended: 20-80)
- **Top-p**: Use nucleus sampling for natural responses (recommended: 0.8-0.95)
- **Max Tokens**: Control response length (default: 20)

### Quick Presets
- **Conservative**: Temperature=0.3, Top-k=20, Top-p=0.8
- **Balanced**: Temperature=1.0, Top-k=50, Top-p=0.9
- **Creative**: Temperature=1.5, Top-k=80, Top-p=0.95
- **Random**: Temperature=1.8, Top-k=100, Top-p=0.98

##  Model Architecture

### Transformer Components
```python
# Model Configuration
context_length = 32      # Sequence length
vocab_size = 10500       # Vocabulary size
embedding_dim = 12       # Hidden dimension
num_heads = 4            # Attention heads
num_layers = 8           # Transformer layers
```

### Key Features
- **Causal Attention**: Ensures autoregressive generation
- **Rotary Position Encoding**: Better positional understanding
- **Custom MLP**: Gated linear units with GELU activation
- **Efficient Design**: Optimized for CPU inference

##  UI Customization

The interface features a modern design system with:
- **CSS Custom Properties**: Easy theme customization
- **Responsive Breakpoints**: Mobile-first design
- **Animation System**: Smooth transitions and micro-interactions
- **Component Library**: Reusable UI elements

### Color Palette
```css
--primary-green: #00ff88      /* Neon green primary */
--primary-cyan: #00ffff       /* Cyan accents */
--background-main: #000000    /* Pure black background */
--text-primary: #ffffff       /* High contrast text */
```

## Performance

- **Model Size**: ~1.1MB (lightweight and portable)
- **Inference Speed**: Real-time on modern CPUs
- **Memory Usage**: Low memory footprint
- **Vocabulary**: 10,500+ tokens with BPE encoding

## 🔧 Advanced Configuration

### Custom Model Loading
```python
# Load model from URL
model_url = "https://example.com/custom_model.pth"

# Load local model
model_path = "path/to/your/model.pth"
```

### Sampling Parameters
```python
# Fine-tune generation
temperature = 1.0     # Creativity control
top_k = 50           # Vocabulary limitation
top_p = 0.9          # Nucleus sampling
max_tokens = 20      # Response length
```

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **PyTorch**: Deep learning framework
- **Gradio**: Web interface framework
- **Transformers**: Tokenizer implementation
- **Modern CSS**: Design inspiration from contemporary web standards

##  Roadmap

- [ ] GPU acceleration support
- [ ] Larger model variants
- [ ] Multi-language support
- [ ] Advanced fine-tuning capabilities
- [ ] API endpoint integration
- [ ] Model quantization options

---

** Star this repository if you found it helpful!**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.33+-orange.svg)
