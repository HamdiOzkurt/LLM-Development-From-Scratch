import os
from gradio.components import clear_button
import torch
import torch.nn.functional as F
from version_v1.master_model import MasterModel as UstaModel
from version_v1.master_tokenizer import MasterTokenizer as UstaTokenizer
import gradio as gr

model, tokenizer, model_status = None, None, "Not Loaded"

def load_model(custom_model_path=None):
    try:
        # Tokenizer yükleme
        u_tokenizer = UstaTokenizer.from_pretrained("version_v1/my_bpe_tokenizer.json")
        print(f"✅ Tokenizer loaded successfully, vocab_size: {len(u_tokenizer.vocab)}")

        # Model parametreleri
        context_length = 32
        vocab_size = len(u_tokenizer.vocab)  # 10500
        embedding_dim = 12
        num_heads = 4
        num_layers = 8

        print(f"🏗️ Creating model with vocab_size: {vocab_size}")
        
        model = UstaModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            num_layers=num_layers
        )
        
        # CUDA sorunu çözümü - CPU'ya map et
        model_path = custom_model_path if (custom_model_path and os.path.exists(custom_model_path)) else "version_v1/u_model.pth"
        
        print(f"📁 Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        return model, u_tokenizer, "✅ Model loaded successfully"
    
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return None, None, error_msg

# Gelişmiş sampling fonksiyonu
def sample_with_params(logits, temperature=1.0, top_k=None, top_p=None):
    """Gelişmiş sampling parametreleri ile token seçimi"""
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
    
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# Model başlatma - DEBUG bilgileriyle
print("🚀 Initializing model...")
print(f"🖥️ CUDA available: {torch.cuda.is_available()}")
print(f"📁 Working directory: {os.getcwd()}")

try:
    model, tokenizer, model_status = load_model(None)
    print(f"📊 Final status: {model_status}")
    
    # Debug: Model ve tokenizer kontrolü
    print(f"🔍 Model is None: {model is None}")
    print(f"🔍 Tokenizer is None: {tokenizer is None}")
    
except Exception as e:
    print(f"❌ Critical error: {e}")
    model, tokenizer, model_status = None, None, f"Critical error: {str(e)}"

def chat_with_model(message, chat_history, max_new_tokens=20, temperature=1.0, top_k=50, top_p=0.9):
    global model, tokenizer
    
    # Güvenlik kontrolleri
    if tokenizer is None:
        print("❌ Tokenizer is None!")
        chat_history.append([message, "❌ Tokenizer not loaded. Please reload model."])
        return chat_history, ""
        
    if model is None:
        print("❌ Model is None!")  
        chat_history.append([message, "❌ Model not loaded. Please reload model."])
        return chat_history, ""
    

def generate_reply_text(message: str, max_new_tokens=20, temperature=1.0, top_k=50, top_p=0.9) -> str:
    """Modelden yalnızca cevap metnini üretir (chat geçmişine dokunmaz)."""
    global model, tokenizer
    if tokenizer is None or model is None:
        return "Model hazır değil. Lütfen modeli yeniden yükleyin."

    try:
        tokens = tokenizer.encode(message)
        if len(tokens) > 25:
            tokens = tokens[-25:]

        with torch.no_grad():
            actual_max_tokens = min(max_new_tokens, 32 - len(tokens))
            input_ids = torch.tensor(tokens).unsqueeze(0) if not isinstance(tokens, torch.Tensor) else tokens.detach().clone().unsqueeze(0)

            for _ in range(actual_max_tokens):
                if input_ids.size(1) >= 32:
                    break
                outputs = model(input_ids)
                next_token_logits = outputs[0, -1, :]
                next_token = sample_with_params(
                    next_token_logits.unsqueeze(0),
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                )
                input_ids = torch.cat([input_ids, next_token], dim=1)

        full_response = tokenizer.decode(input_ids.squeeze().tolist())
        original_message = tokenizer.decode(tokens)
        response = full_response[len(original_message):] if full_response.startswith(original_message) else full_response
        response = response.replace("<pad>", "").replace("<unk>", "").strip()
        return response if len(response) > 0 else "Anlamadım"
    except Exception as e:
        return f"❌ Generation error: {str(e)}"
    
    try:
        # Encoding
        tokens = tokenizer.encode(message)
        if len(tokens) > 25:
            tokens = tokens[-25:]
        
        print(f"🔤 Encoded tokens: {len(tokens)}")
        
        with torch.no_grad():
            actual_max_tokens = min(max_new_tokens, 32 - len(tokens))
            
            if isinstance(tokens, torch.Tensor):
                input_ids = tokens.detach().clone().unsqueeze(0)
            else:
                input_ids = torch.tensor(tokens).unsqueeze(0)
            
            for _ in range(actual_max_tokens):
                if input_ids.size(1) >= 32:
                    break
                    
                outputs = model(input_ids)
                next_token_logits = outputs[0, -1, :]
                
                next_token = sample_with_params(
                    next_token_logits.unsqueeze(0), 
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None
                )
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decoding
        full_response = tokenizer.decode(input_ids.squeeze().tolist())
        original_message = tokenizer.decode(tokens)
        
        if full_response.startswith(original_message):
            response = full_response[len(original_message):] 
        else:
            response = full_response

        response = response.replace("<pad>","").replace("<unk>","").strip()

        if len(response) <= 0:
            response = "I don't understand"
        
        chat_history.append([message, response])
        return chat_history, ""
    
    except Exception as e:
        error_msg = f"❌ Generation error: {str(e)}"
        print(error_msg)
        chat_history.append([message, error_msg])
        return chat_history, ""

def reload_model():
    """Manuel model yeniden yükleme"""
    global model, tokenizer, model_status
    print("🔄 Manually reloading model...")
    model, tokenizer, model_status = load_model(None)
    return model_status

def load_model_from_url(custom_model_url):
    global model, tokenizer, model_status
    try:
        if not custom_model_url.strip():
            # Boş URL - yerel modeli yeniden yükle
            model, tokenizer, model_status = load_model(None)
            return model_status
            
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(custom_model_url, headers=headers)
        response.raise_for_status()

        temp_file = "temp_model.pth"
        with open(temp_file, "wb") as f:
            f.write(response.content)

        model, tokenizer, model_status = load_model(temp_file)
        os.remove(temp_file)
        return model_status
    except Exception as e:
        error_msg = f"❌ URL load error: {str(e)}"
        print(error_msg)
        return error_msg

# Ultra Modern Professional CSS styling with Enhanced Dark/Light Theme Support
modern_css = """
/* Enhanced Modern Professional Color System */
:root {
    /* Primary Brand Colors - Neon Green/Cyan Theme */
    --primary-green: #00ff88;
    --primary-cyan: #00ffff;
    --primary-blue: #0099ff;
    --primary-teal: #14b8a6;
    
    /* Advanced Gradients - Neon Theme */
    --primary-gradient: linear-gradient(135deg, #00ff88 0%, #00ffcc 50%, #00ffff 100%);
    --secondary-gradient: linear-gradient(135deg, #0099ff 0%, #00ccff 100%);
    --accent-gradient: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
    --success-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
    --danger-gradient: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    --warning-gradient: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    
    /* Dark Theme Variables - Pure Black Background */
    --background-main: #000000;
    --background-secondary: #0a0a0a;
    --background-tertiary: #151515;
    --background-card: rgba(10, 10, 10, 0.95);
    --background-card-hover: rgba(20, 20, 20, 0.95);
    --background-input: rgba(25, 25, 25, 0.9);
    --background-input-focus: rgba(35, 35, 35, 0.95);
    --background-glass: rgba(15, 15, 15, 0.8);
    --background-overlay: rgba(0, 0, 0, 0.9);
    
    --text-primary: #ffffff;
    --text-secondary: #e5e5e5;
    --text-muted: #999999;
    --text-accent: #00ffcc;
    --text-success: #00ff88;
    --text-danger: #ff4444;
    --text-warning: #ffaa00;
    
    --border-subtle: rgba(255, 255, 255, 0.1);
    --border-focus: rgba(0, 255, 204, 0.5);
    --border-accent: rgba(0, 255, 136, 0.4);
    --border-success: rgba(0, 255, 136, 0.4);
    --border-danger: rgba(255, 68, 68, 0.4);
    
    /* Enhanced Shadow System - Neon Glow */
    --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px 0 rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -2px rgba(0, 0, 0, 0.4);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.7), 0 10px 10px -5px rgba(0, 0, 0, 0.5);
    --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    --shadow-glow: 0 0 20px rgba(0, 255, 204, 0.4);
    --shadow-glow-lg: 0 0 40px rgba(0, 255, 204, 0.6);
    --shadow-inner: inset 0 2px 4px 0 rgba(0, 0, 0, 0.4);
    
    /* Animation Variables */
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
    --bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);
    
    /* Spacing System */
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    --space-2xl: 3rem;
    
    /* Border Radius System */
    --radius-xs: 0.25rem;
    --radius-sm: 0.5rem;
    --radius-md: 0.75rem;
    --radius-lg: 1rem;
    --radius-xl: 1.5rem;
    --radius-2xl: 2rem;
    --radius-full: 9999px;
}

/* Enhanced Global Styles */
* {
    transition: all var(--transition-normal) !important;
}

*:focus {
    outline: 2px solid var(--border-focus) !important;
    outline-offset: 2px !important;
}

html {
    scroll-behavior: smooth !important;
    font-size: 16px !important;
}

body, .gradio-container {
    background: var(--background-main) !important;
    background-image: 
        radial-gradient(circle at 25% 25%, hsla(160, 100%, 50%, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 75% 25%, hsla(180, 100%, 50%, 0.06) 0%, transparent 50%),
        radial-gradient(circle at 50% 75%, hsla(200, 100%, 50%, 0.04) 0%, transparent 50%),
        linear-gradient(135deg, hsla(160, 100%, 50%, 0.03) 0%, transparent 100%) !important;
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif !important;
    color: var(--text-primary) !important;
    line-height: 1.7 !important;
    min-height: 100vh !important;
    position: relative !important;
    overflow-x: hidden !important;
}

/* Enhanced Theme Toggle Support */
@media (prefers-color-scheme: light) {
    :root {
        /* Light Theme Color System */
        --background-main: #fafafa;
        --background-secondary: #ffffff;
        --background-tertiary: #f5f5f5;
        --background-card: rgba(255, 255, 255, 0.95);
        --background-card-hover: rgba(248, 250, 252, 0.95);
        --background-input: rgba(248, 250, 252, 0.8);
        --background-input-focus: rgba(241, 245, 249, 0.9);
        --background-glass: rgba(255, 255, 255, 0.7);
        --background-overlay: rgba(0, 0, 0, 0.6);

        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --text-accent: #6366f1;
        --text-success: #059669;
        --text-danger: #dc2626;
        --text-warning: #d97706;

        --border-subtle: rgba(15, 23, 42, 0.08);
        --border-focus: rgba(99, 102, 241, 0.4);
        --border-accent: rgba(99, 102, 241, 0.3);
        --border-success: rgba(5, 150, 105, 0.3);
        --border-danger: rgba(220, 38, 38, 0.3);

        --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
        --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.2);
        --shadow-glow-lg: 0 0 40px rgba(99, 102, 241, 0.3);
        --shadow-inner: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);
    }

    body, .gradio-container {
        background: var(--background-main) !important;
        background-image: 
            radial-gradient(circle at 25% 25%, hsla(258, 100%, 75%, 0.02) 0%, transparent 50%),
            radial-gradient(circle at 75% 25%, hsla(230, 100%, 75%, 0.02) 0%, transparent 50%),
            radial-gradient(circle at 50% 75%, hsla(340, 100%, 75%, 0.02) 0%, transparent 50%),
            linear-gradient(135deg, hsla(258, 100%, 75%, 0.01) 0%, transparent 100%) !important;
        color: var(--text-primary) !important;
    }
}

/* Modern Header Design */
.gradio-container h1:first-of-type {
    background: var(--primary-gradient) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 800 !important;
    font-size: clamp(2rem, 5vw, 3rem) !important;
    margin-bottom: var(--space-sm) !important;
    text-align: center !important;
    letter-spacing: -0.025em !important;
    position: relative !important;
    z-index: 1 !important;
}

.gradio-container .markdown:first-of-type + .markdown p {
    color: var(--text-secondary) !important;
    font-size: clamp(0.9rem, 2vw, 1.1rem) !important;
    margin-bottom: var(--space-2xl) !important;
    text-align: center !important;
    font-weight: 400 !important;
    max-width: 700px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Enhanced Main Container */
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
    padding: var(--space-xl) !important;
    position: relative !important;
}

.gradio-container::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    background: radial-gradient(circle at 50% 0%, rgba(0, 255, 204, 0.05) 0%, transparent 70%) !important;
    pointer-events: none !important;
    z-index: 0 !important;
}

/* Modern Chat Container */
.gradio-row > .gradio-column:first-child {
    background: var(--background-card) !important;
    border-radius: var(--radius-2xl) !important;
    padding: var(--space-xl) !important;
    margin-right: var(--space-lg) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-2xl) !important;
    position: relative !important;
    overflow: hidden !important;
    z-index: 1 !important;
}

.gradio-row > .gradio-column:first-child::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 2px !important;
    background: var(--primary-gradient) !important;
    opacity: 0.8 !important;
    border-radius: var(--radius-2xl) var(--radius-2xl) 0 0 !important;
}

.gradio-row > .gradio-column:first-child::after {
    content: '' !important;
    position: absolute !important;
    top: -50% !important;
    left: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background: radial-gradient(circle, rgba(0, 255, 136, 0.04) 0%, transparent 70%) !important;
    animation: float 6s ease-in-out infinite !important;
    pointer-events: none !important;
    z-index: -1 !important;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(-2px, -2px) rotate(1deg); }
}

/* Chat Interface */
.chatbot {
    background: transparent !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
    min-height: 400px !important;
}

/* Modern Message Input */
.textbox input, .textbox textarea {
    background: var(--background-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-xl) !important;
    color: var(--text-primary) !important;
    padding: var(--space-md) var(--space-lg) !important;
    font-size: 1rem !important;
    backdrop-filter: blur(12px) saturate(150%) !important;
    font-weight: 400 !important;
    transition: all var(--transition-normal) !important;
    box-shadow: var(--shadow-sm) !important;
    position: relative !important;
    z-index: 1 !important;
}

.textbox input:focus, .textbox textarea:focus {
    border-color: var(--primary-green) !important;
    box-shadow: 0 0 0 3px var(--border-focus), var(--shadow-glow) !important;
    outline: none !important;
    background: var(--background-input-focus) !important;
    transform: translateY(-2px) scale(1.005) !important;
}

.textbox input:hover, .textbox textarea:hover {
    border-color: var(--border-accent) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

.textbox input::placeholder, .textbox textarea::placeholder {
    color: var(--text-muted) !important;
    font-weight: 400 !important;
    transition: opacity var(--transition-fast) !important;
}

.textbox input:focus::placeholder, .textbox textarea:focus::placeholder {
    opacity: 0.7 !important;
}

/* Modern Enhanced Buttons */
.gradio-button.primary {
    background: var(--primary-gradient) !important;
    border: none !important;
    border-radius: var(--radius-xl) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: var(--space-md) var(--space-xl) !important;
    font-size: 1rem !important;
    box-shadow: var(--shadow-lg) !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all var(--transition-normal) !important;
    cursor: pointer !important;
    user-select: none !important;
}

.gradio-button.primary::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent) !important;
    transition: left var(--transition-slow) !important;
}

.gradio-button.primary:hover::before {
    left: 100% !important;
}

.gradio-button.primary:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: var(--shadow-2xl) !important;
}

.gradio-button.primary:active {
    transform: translateY(-1px) scale(0.98) !important;
    transition: all var(--transition-fast) !important;
}

.gradio-button.secondary {
    background: var(--background-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-xl) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: var(--space-md) var(--space-xl) !important;
    font-size: 0.95rem !important;
    backdrop-filter: blur(12px) saturate(150%) !important;
    transition: all var(--transition-normal) !important;
    cursor: pointer !important;
    user-select: none !important;
    position: relative !important;
    overflow: hidden !important;
}

.gradio-button.secondary::before {
    content: '' !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    width: 0 !important;
    height: 0 !important;
    background: radial-gradient(circle, var(--primary-green), transparent) !important;
    opacity: 0.1 !important;
    transform: translate(-50%, -50%) !important;
    transition: all var(--transition-normal) !important;
    border-radius: 50% !important;
}

.gradio-button.secondary:hover::before {
    width: 200px !important;
    height: 200px !important;
}

.gradio-button.secondary:hover {
    background: var(--background-card-hover) !important;
    border-color: var(--border-accent) !important;
    color: var(--text-primary) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: var(--shadow-lg) !important;
}

.gradio-button.secondary:active {
    transform: translateY(0) scale(0.98) !important;
    transition: all var(--transition-fast) !important;
}

/* Modern Settings Panel */
.gradio-row > .gradio-column:last-child {
    background: var(--background-card) !important;
    border-radius: var(--radius-2xl) !important;
    padding: var(--space-xl) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-2xl) !important;
    min-width: 380px !important;
    position: relative !important;
    overflow: hidden !important;
    z-index: 1 !important;
}

.gradio-row > .gradio-column:last-child::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 2px !important;
    background: var(--secondary-gradient) !important;
    opacity: 0.8 !important;
    border-radius: var(--radius-2xl) var(--radius-2xl) 0 0 !important;
}

.gradio-row > .gradio-column:last-child::after {
    content: '' !important;
    position: absolute !important;
    top: -50% !important;
    right: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background: radial-gradient(circle, rgba(0, 255, 255, 0.03) 0%, transparent 70%) !important;
    animation: float-reverse 8s ease-in-out infinite !important;
    pointer-events: none !important;
    z-index: -1 !important;
}

@keyframes float-reverse {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(2px, -2px) rotate(-1deg); }
}

/* Settings Title */
.gradio-column h2 {
    color: var(--text-primary) !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    margin-bottom: 1.5rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    letter-spacing: -0.01em !important;
}

/* Enhanced Sliders */
.gradio-slider {
    margin-bottom: 2rem !important;
    padding: 1rem !important;
    background: rgba(55, 65, 81, 0.3) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
}

.gradio-slider label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}

.gradio-slider input[type="number"] {
    background: var(--background-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-accent) !important;
    padding: 0.5rem !important;
    font-size: 0.8rem !important;
    width: 70px !important;
    text-align: center !important;
    font-weight: 600 !important;
}

.gradio-slider input[type="range"] {
    background: transparent !important;
    height: 8px !important;
    border-radius: 4px !important;
    outline: none !important;
    -webkit-appearance: none !important;
}

.gradio-slider input[type="range"]::-webkit-slider-track {
    background: rgba(209, 213, 219, 0.2) !important;
    height: 8px !important;
    border-radius: 4px !important;
}

.gradio-slider input[type="range"]::-webkit-slider-thumb {
    background: var(--primary-gradient) !important;
    width: 22px !important;
    height: 22px !important;
    border-radius: 50% !important;
    border: 3px solid white !important;
    box-shadow: var(--shadow-md) !important;
    cursor: pointer !important;
    -webkit-appearance: none !important;
    transition: all 0.2s ease !important;
}

.gradio-slider input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.1) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* Model Management Section */
.gradio-container > div:last-child {
    margin-top: 3rem !important;
    padding-top: 2rem !important;
    border-top: 1px solid var(--border-subtle) !important;
}

.gradio-row:has(.gradio-textbox) .gradio-column {
    background: var(--background-card) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 0 0.5rem !important;
    border: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: var(--shadow-md) !important;
}

.gradio-row:has(.gradio-textbox) .gradio-column:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* Status Display */
.textbox[readonly] input {
    background: rgba(0, 255, 136, 0.15) !important;
    border-color: var(--primary-green) !important;
    color: var(--primary-green) !important;
    font-weight: 600 !important;
    text-align: center !important;
}

/* Features List */
.gradio-container .markdown ul {
    list-style: none !important;
    padding: 0 !important;
    display: grid !important;
    gap: 0.75rem !important;
}

.gradio-container .markdown ul li {
    background: var(--background-card) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    border-left: 4px solid var(--primary-purple) !important;
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--border-subtle) !important;
    transition: all 0.3s ease !important;
}

.gradio-container .markdown ul li:hover {
    transform: translateX(4px) !important;
    background: var(--background-card-hover) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Professional Avatar Styling */
.chatbot .user-message img,
.chatbot .bot-message img,
.chatbot img[alt="User"],
.chatbot img[alt="Assistant"] {
    width: 48px !important;
    height: 48px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    filter: drop-shadow(0 4px 12px rgba(139, 92, 246, 0.25)) !important;
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
    position: relative !important;
}

/* User Avatar Styling */
.chatbot .user-message img,
.chatbot img[alt="User"] {
    filter: drop-shadow(0 4px 12px rgba(102, 126, 234, 0.3)) !important;
    border: 2px solid rgba(102, 126, 234, 0.2) !important;
}

/* AI Avatar Styling */
.chatbot .bot-message img,
.chatbot img[alt="Assistant"] {
    filter: drop-shadow(0 4px 12px rgba(139, 92, 246, 0.4)) !important;
    border: 2px solid rgba(139, 92, 246, 0.2) !important;
    animation: ai-pulse 4s ease-in-out infinite !important;
}

/* Avatar Hover Effects */
.chatbot .user-message img:hover,
.chatbot .bot-message img:hover,
.chatbot img[alt="User"]:hover,
.chatbot img[alt="Assistant"]:hover {
    transform: scale(1.1) rotate(5deg) !important;
    filter: drop-shadow(0 8px 20px rgba(139, 92, 246, 0.5)) brightness(1.1) !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
}

/* AI Avatar Animation */
@keyframes ai-pulse {
    0%, 100% {
        filter: drop-shadow(0 4px 12px rgba(139, 92, 246, 0.4)) !important;
        transform: scale(1) !important;
    }
    50% {
        filter: drop-shadow(0 6px 16px rgba(139, 92, 246, 0.6)) !important;
        transform: scale(1.05) !important;
    }
}

/* Professional Chat Messages - Asymmetric Design */
.chatbot .message {
    margin: 8px 0 !important;
    max-width: 85% !important;
    clear: both !important;
    position: relative !important;
    display: block !important;
    animation: messageSlideIn 0.3s ease-out !important;
}

/* User Messages - Right Aligned */
.chatbot .message:nth-child(odd) {
    float: right !important;
    margin-left: auto !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.25) !important;
    border: none !important;
}

/* AI Messages - Left Aligned */
.chatbot .message:nth-child(even) {
    float: left !important;
    margin-right: auto !important;
    background: rgba(55, 65, 81, 0.8) !important;
    color: var(--text-primary) !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
}

/* Message Text Styling */
.chatbot .message p {
    margin: 0 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    font-weight: 400 !important;
    word-wrap: break-word !important;
}

/* User message text color override */
.chatbot .message:nth-child(odd) p {
    color: white !important;
}

/* Message Hover Effects */
.chatbot .message:hover {
    transform: translateY(-1px) !important;
    transition: all 0.2s ease !important;
}

.chatbot .message:nth-child(odd):hover {
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35) !important;
}

.chatbot .message:nth-child(even):hover {
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25) !important;
}

/* Message Animation */
@keyframes messageSlideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Clear float issues */
.chatbot::after {
    content: "" !important;
    display: table !important;
    clear: both !important;
}

/* Enhanced Chat Layout */
.chatbot {
    background: transparent !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 16px 12px !important;
    margin-bottom: 1.5rem !important;
    min-height: 450px !important;
    max-height: 500px !important;
    overflow-y: auto !important;
    scroll-behavior: smooth !important;
}

/* Chat welcome message styling */
.chatbot .placeholder {
    text-align: center !important;
    color: var(--text-muted) !important;
    font-style: italic !important;
    margin: 2rem 0 !important;
    padding: 2rem !important;
    background: rgba(139, 92, 246, 0.05) !important;
    border-radius: 16px !important;
    border: 1px dashed rgba(139, 92, 246, 0.2) !important;
}

/* Chat Scrollbar Styling */
.chatbot::-webkit-scrollbar {
    width: 6px !important;
}

.chatbot::-webkit-scrollbar-track {
    background: rgba(55, 65, 81, 0.2) !important;
    border-radius: 3px !important;
}

.chatbot::-webkit-scrollbar-thumb {
    background: var(--primary-gradient) !important;
    border-radius: 3px !important;
    opacity: 0.7 !important;
}

.chatbot::-webkit-scrollbar-thumb:hover {
    opacity: 1 !important;
}

/* Hide default avatar containers/images in chatbot bubbles */
.chatbot img[alt="User"],
.chatbot img[alt="Assistant"],
.chatbot .avatar,
.chatbot .avatar-container {
    display: none !important;
}

/* Quick Presets segmented control */
.quick-presets .wrap {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 8px !important;
}

.quick-presets input[type="radio"] + label {
    background: rgba(55, 65, 81, 0.35) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    color: var(--text-secondary) !important;
}

.quick-presets input[type="radio"]:checked + label {
    background: var(--primary-gradient) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: var(--shadow-md) !important;
}


/* Ultra Modern Responsive Design */
/* Tablet & Small Desktop */
@media (max-width: 1200px) {
    .gradio-container {
        max-width: 100% !important;
        padding: var(--space-lg) !important;
    }
    
    .gradio-row > .gradio-column:last-child {
        min-width: 340px !important;
    }
}

@media (max-width: 1024px) {
    .gradio-container {
        padding: var(--space-md) !important;
    }
    
    .gradio-row {
        flex-direction: column !important;
        gap: var(--space-lg) !important;
    }
    
    .gradio-row > .gradio-column {
        margin: 0 !important;
        min-width: unset !important;
        width: 100% !important;
    }
    
    .gradio-row > .gradio-column:first-child {
        margin-right: 0 !important;
        order: 2 !important;
    }
    
    .gradio-row > .gradio-column:last-child {
        order: 1 !important;
        min-width: unset !important;
    }
    
    .gradio-container h1:first-of-type {
        font-size: clamp(1.75rem, 4vw, 2.5rem) !important;
    }
    
    .chatbot {
        min-height: 350px !important;
        max-height: 400px !important;
    }
}

/* Mobile Landscape */
@media (max-width: 768px) {
    .gradio-container {
        padding: var(--space-sm) !important;
    }
    
    .gradio-row > .gradio-column:first-child,
    .gradio-row > .gradio-column:last-child {
        padding: var(--space-lg) !important;
        border-radius: var(--radius-xl) !important;
    }
    
    .chatbot {
        min-height: 300px !important;
        max-height: 350px !important;
    }
    
    .textbox input, .textbox textarea {
        padding: var(--space-sm) var(--space-md) !important;
        font-size: 0.95rem !important;
    }
    
    .gradio-button.primary, .gradio-button.secondary {
        padding: var(--space-sm) var(--space-md) !important;
        font-size: 0.9rem !important;
    }
}

/* Mobile Portrait */
@media (max-width: 640px) {
    .gradio-container {
        padding: var(--space-xs) !important;
    }
    
    .gradio-row > .gradio-column:first-child,
    .gradio-row > .gradio-column:last-child {
        padding: var(--space-md) !important;
        border-radius: var(--radius-lg) !important;
    }
    
    .gradio-container h1:first-of-type {
        font-size: clamp(1.5rem, 6vw, 2rem) !important;
        margin-bottom: var(--space-xs) !important;
    }
    
    .gradio-container .markdown:first-of-type + .markdown p {
        font-size: clamp(0.85rem, 3vw, 1rem) !important;
        margin-bottom: var(--space-lg) !important;
    }
    
    .chatbot {
        min-height: 280px !important;
        max-height: 320px !important;
        padding: var(--space-sm) !important;
    }
    
    .textbox input, .textbox textarea {
        padding: var(--space-sm) !important;
        font-size: 0.9rem !important;
        border-radius: var(--radius-md) !important;
    }
    
    .gradio-button.primary, .gradio-button.secondary {
        padding: var(--space-sm) !important;
        font-size: 0.85rem !important;
        border-radius: var(--radius-md) !important;
    }
    
    .gradio-slider {
        margin-bottom: var(--space-md) !important;
        padding: var(--space-sm) !important;
    }
    
    .chatbot .message {
        max-width: 95% !important;
        padding: var(--space-sm) var(--space-md) !important;
        font-size: 0.9rem !important;
    }
}

/* Small Mobile */
@media (max-width: 480px) {
    .gradio-container::before {
        display: none !important;
    }
    
    .gradio-row > .gradio-column:first-child::after,
    .gradio-row > .gradio-column:last-child::after {
        display: none !important;
    }
    
    .chatbot .message {
        border-radius: var(--radius-md) var(--radius-md) var(--radius-xs) var(--radius-md) !important;
    }
    
    .chatbot .message:nth-child(even) {
        border-radius: var(--radius-md) var(--radius-md) var(--radius-md) var(--radius-xs) !important;
    }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px !important;
}

::-webkit-scrollbar-track {
    background: rgba(55, 65, 81, 0.3) !important;
    border-radius: 4px !important;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-gradient) !important;
    border-radius: 4px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-gradient) !important;
}

/* Advanced Loading and Micro-interactions */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes shimmer {
    0% { background-position: -200px 0; }
    100% { background-position: calc(200px + 100%) 0; }
}

@keyframes glow {
    0%, 100% { 
        box-shadow: 0 0 5px rgba(0, 255, 204, 0.3);
    }
    50% { 
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.6), 0 0 30px rgba(0, 255, 136, 0.2);
    }
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes bounce-in {
    0% {
        opacity: 0;
        transform: scale(0.3) translateY(50px);
    }
    50% {
        opacity: 1;
        transform: scale(1.05) translateY(-10px);
    }
    70% {
        transform: scale(0.9) translateY(0);
    }
    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

.loading {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite !important;
}

.shimmer {
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent) !important;
    background-size: 200px 100% !important;
    animation: shimmer 2s infinite !important;
}

/* Enhanced Component Animations */
.gradio-row > .gradio-column:first-child {
    animation: slideInLeft 0.6s ease-out !important;
}

.gradio-row > .gradio-column:last-child {
    animation: slideInRight 0.6s ease-out 0.2s both !important;
}

.gradio-container h1:first-of-type {
    animation: slideInUp 0.8s ease-out !important;
}

.gradio-container .markdown:first-of-type + .markdown p {
    animation: slideInUp 0.8s ease-out 0.3s both !important;
}

/* Interactive Hover Effects */
.gradio-row > .gradio-column:first-child:hover,
.gradio-row > .gradio-column:last-child:hover {
    transform: translateY(-4px) !important;
    box-shadow: var(--shadow-2xl) !important;
}

/* Focus Ring Enhancement */
.gradio-button:focus-visible,
.textbox input:focus-visible,
.textbox textarea:focus-visible {
    animation: glow 2s ease-in-out infinite !important;
}

/* Status Indicator Animations */
.textbox[readonly] input {
    position: relative !important;
    overflow: hidden !important;
}

.textbox[readonly] input::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.2), transparent) !important;
    animation: shimmer 2s infinite !important;
}

/* Markdown and code blocks styling inside chatbot */
.chatbot pre, .chatbot code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
}

.chatbot pre {
    background: #0b1220 !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    color: #e5e7eb !important;
    padding: 12px 14px !important;
    border-radius: 10px !important;
    overflow: auto !important;
}

.chatbot code:not(pre code) {
    background: rgba(167, 139, 250, 0.1) !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    color: #e9d5ff !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
}

/* Typing indicator */
.typing-indicator {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    color: #c4b5fd !important;
}

.typing-indicator .dot {
    width: 6px !important;
    height: 6px !important;
    border-radius: 50% !important;
    background: #c4b5fd !important;
    animation: typingBlink 1.2s infinite ease-in-out !important;
}

.typing-indicator .dot:nth-child(2) { animation-delay: 0.15s !important; }
.typing-indicator .dot:nth-child(3) { animation-delay: 0.3s !important; }

@keyframes typingBlink {
    0%, 80%, 100% { opacity: 0.2; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-2px); }
}

/* Enhanced Light Theme - Better integrated with animations */
@media (prefers-color-scheme: light) {
    .gradio-container .markdown:first-of-type + .markdown p {
        color: var(--text-secondary) !important;
    }

    .gradio-row > .gradio-column:first-child,
    .gradio-row > .gradio-column:last-child {
        background: var(--background-card) !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: var(--shadow-xl) !important;
    }

    .textbox input, .textbox textarea {
        background: var(--background-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    .gradio-button.secondary {
        background: var(--background-card) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    .gradio-slider label { color: var(--text-primary) !important; }

    .gradio-slider input[type="number"] {
        background: var(--background-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    .gradio-slider input[type="range"]::-webkit-slider-track {
        background: var(--border-subtle) !important;
    }

    /* Chat Messages - Light Theme */
    .chatbot {
        background: var(--background-card) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    /* User Messages - Light theme adjustment */
    .chatbot .message:nth-child(odd) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* AI Messages - Light theme adjustment */
    .chatbot .message:nth-child(even) {
        background: var(--background-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* Message hover effects - Light theme */
    .chatbot .message:nth-child(odd):hover {
        box-shadow: var(--shadow-lg) !important;
    }

    .chatbot .message:nth-child(even):hover {
        box-shadow: var(--shadow-md) !important;
        background: var(--background-card-hover) !important;
    }

    /* Code blocks readable on light */
    .chatbot pre {
        background: var(--background-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    .chatbot code:not(pre code) {
        background: rgba(99, 102, 241, 0.08) !important;
        color: var(--text-accent) !important;
        border: 1px solid var(--border-accent) !important;
    }

    /* Quick presets labels visibility */
    .quick-presets input[type="radio"] + label {
        background: var(--background-card) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    /* Light theme specific animations */
    .gradio-row > .gradio-column:first-child::after,
    .gradio-row > .gradio-column:last-child::after {
        background: radial-gradient(circle, rgba(99, 102, 241, 0.015) 0%, transparent 70%) !important;
    }

    .gradio-container::before {
        background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.02) 0%, transparent 70%) !important;
    }
}
"""

# Professional Gradio interface with modern design
with gr.Blocks(
    title="🚀 Master LLM Chat - AI Conversation Platform", 
    css=modern_css, 
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    analytics_enabled=False
) as demo:
    
    # Header Section with Enhanced Branding
    with gr.Column(elem_classes="header-section"):
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="
                background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #c084fc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 3rem;
                font-weight: 800;
                margin: 0;
                letter-spacing: -0.02em;
                line-height: 1.1;
            ">🚀 Master LLM Chat</h1>
            <p style="
                color: #d1d5db;
                font-size: 1.1rem;
                margin: 1rem 0;
                font-weight: 400;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
            ">✨ Experience the future of AI conversation with our custom transformer language model featuring cutting-edge neural architecture</p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 1rem;
                margin-top: 1.5rem;
                flex-wrap: wrap;
            ">
                <span style="
                    background: rgba(139, 92, 246, 0.1);
                    color: #a78bfa;
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 500;
                    border: 1px solid rgba(167, 139, 250, 0.2);
                ">🧠 Advanced Neural Architecture</span>
                <span style="
                    background: rgba(59, 130, 246, 0.1);
                    color: #60a5fa;
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 500;
                    border: 1px solid rgba(96, 165, 250, 0.2);
                ">⚡ Real-time Inference</span>
                <span style="
                    background: rgba(16, 185, 129, 0.1);
                    color: #10b981;
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 500;
                    border: 1px solid rgba(16, 185, 129, 0.2);
                ">🎯 Custom Tokenization</span>
            </div>
        </div>
        """)
    
    # Main Chat Interface
    with gr.Row(equal_height=True):
        # Chat Column - Enhanced
        with gr.Column(scale=3, elem_classes="chat-column"):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                container=False,
                bubble_full_width=False,
                show_copy_button=True,
                placeholder="<div style='text-align: center; color: #9ca3af; font-size: 1.1rem; padding: 2rem;'>💬 AI asistanınızla konuşmaya başlayın!<br><br>🚀 Aşağıya bir mesaj yazarak başlayalım...</div>"
            )
            
            # Message Input Area - Enhanced
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="💭 Bana her şeyi sorabilirsin... Hadi birlikte imkânları keşfedelim!", 
                    label="",
                    container=False,
                    scale=5,
                    show_label=False,
                    lines=1,
                    max_lines=3
                )
                send_button = gr.Button("🚀 Send", variant="primary", scale=1, size="lg")

            # Action Buttons
            with gr.Row():
                clear_button = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm", scale=1)
                regenerate_button = gr.Button("🔁 Regenerate", variant="secondary", size="sm", scale=1)
                delete_button = gr.Button("🗑️ Delete Last", variant="secondary", size="sm", scale=1)
                export_button = gr.Button("💾 Export Chat", variant="secondary", size="sm", scale=1)
                copy_button = gr.Button("📋 Copy Last", variant="secondary", size="sm", scale=1)
        
        # Settings Panel - Enhanced
        with gr.Column(scale=1, elem_classes="settings-column"):
            gr.HTML("""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid rgba(209, 213, 219, 0.1);
            ">
                <span style="font-size: 1.5rem;">⚙️</span>
                <h2 style="
                    color: #f9fafb;
                    font-size: 1.3rem;
                    font-weight: 700;
                    margin: 0;
                    letter-spacing: -0.01em;
                ">Generation Settings</h2>
            </div>
            """)

            # Quick Presets - moved to top and redesigned as segmented radio
            quick_preset = gr.Radio(
                ["Conservative", "Balanced", "Creative", "Random"],
                label="Quick Presets",
                value=None,
                interactive=True,
                elem_classes="quick-presets",
            )

            with gr.Group():
                max_new_tokens = gr.Slider(
                    1, 30, value=20, step=1, 
                    label="📝 Max Tokens",
                    info="Maximum number of new tokens to generate",
                    show_label=True,
                    interactive=True
                )
            
            temperature = gr.Slider(
                0.1, 2.0, value=1.0, step=0.1,
                label="🌡️ Temperature", 
                info="Controls randomness in generation",
                    show_label=True,
                    interactive=True
            )
            
            top_k = gr.Slider(
                1, 100, value=50, step=1,
                label="🔝 Top-k",
                info="Limits vocabulary to top-k tokens",
                    show_label=True,
                    interactive=True
            )
            
            top_p = gr.Slider(
                0.1, 1.0, value=0.9, step=0.01,
                label="🎯 Top-p",
                info="Nucleus sampling probability",
                    show_label=True,
                    interactive=True
                )
                # (buttons removed; presets now above as radio)

    # Model Management Section - Enhanced
    gr.HTML("""
    <div style="
        margin: 3rem 0 2rem 0;
        padding: 2rem 0;
        border-top: 1px solid rgba(209, 213, 219, 0.1);
        text-align: center;
    ">
        <h2 style="
            color: #f9fafb;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        ">
            🛠️ Model Management Hub
        </h2>
        <p style="
            color: #9ca3af;
            font-size: 1rem;
            margin: 0;
        ">Manage your AI model configuration and deployment</p>
    </div>
    """)
    
    with gr.Row(equal_height=True):
        # Model Status Panel
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 1rem;
            ">
                <span style="font-size: 1.2rem;">📊</span>
                <h3 style="
                    color: #f9fafb;
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0;
                ">Model Status</h3>
            </div>
            """)
            
            status = gr.Textbox(
                label="", 
                value=model_status, 
                interactive=False,
                show_label=False,
                placeholder="Model status will appear here..."
            )
            
            with gr.Row():
                reload_button = gr.Button("🔄 Reload Local Model", variant="secondary", size="sm", scale=1)
                gr.Button("📈 Model Stats", variant="secondary", size="sm", scale=1, interactive=False)
        
        # Model URL Panel
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 1rem;
            ">
                <span style="font-size: 1.2rem;">🌐</span>
                <h3 style="
                    color: #f9fafb;
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0;
                ">Custom Model URL</h3>
            </div>
            """)
            
            custom_model_url = gr.Textbox(
                placeholder="🌐 https://example.com/model.pth", 
                label="",
                show_label=False,
                lines=1
            )
            
            with gr.Row():
                load_button = gr.Button("📥 Load from URL", variant="primary", size="sm", scale=1)
                gr.Button("🔗 Validate URL", variant="secondary", size="sm", scale=1, interactive=False)

    # Features Showcase - Enhanced
    gr.HTML("""
    <div style="
        margin: 3rem 0 2rem 0;
        padding: 2rem 0;
        border-top: 1px solid rgba(209, 213, 219, 0.1);
    ">
        <h2 style="
            color: #f9fafb;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 2rem;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        ">
            🎨 Platform Features & Capabilities
        </h2>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="
                background: rgba(17, 24, 39, 0.95);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(209, 213, 219, 0.1);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 1rem;
                ">
                    <span style="font-size: 2rem;">🧠</span>
                    <h3 style="
                        color: #f9fafb;
                        font-size: 1.1rem;
                        font-weight: 600;
                        margin: 0;
                    ">Advanced Architecture</h3>
                </div>
                <p style="
                    color: #9ca3af;
                    font-size: 0.9rem;
                    margin: 0;
                    line-height: 1.5;
                ">Custom transformer neural network with multi-head attention mechanisms for superior language understanding.</p>
            </div>
            """)
        
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="
                background: rgba(17, 24, 39, 0.95);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(209, 213, 219, 0.1);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 1rem;
                ">
                    <span style="font-size: 2rem;">🎛️</span>
                    <h3 style="
                        color: #f9fafb;
                        font-size: 1.1rem;
                        font-weight: 600;
                        margin: 0;
                    ">Fine-tuned Sampling</h3>
                </div>
                <p style="
                    color: #9ca3af;
                    font-size: 0.9rem;
                    margin: 0;
                    line-height: 1.5;
                ">Precise control over response generation with Temperature, Top-k, and Top-p sampling techniques.</p>
            </div>
            """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="
                background: rgba(17, 24, 39, 0.95);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(209, 213, 219, 0.1);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 1rem;
                ">
                    <span style="font-size: 2rem;">⚡</span>
                    <h3 style="
                        color: #f9fafb;
                        font-size: 1.1rem;
                        font-weight: 600;
                        margin: 0;
                    ">Optimized Performance</h3>
                </div>
                <p style="
                    color: #9ca3af;
                    font-size: 0.9rem;
                    margin: 0;
                    line-height: 1.5;
                ">CPU-optimized inference engine with intelligent memory management for fast, efficient processing.</p>
            </div>
            """)
        
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="
                background: rgba(17, 24, 39, 0.95);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(209, 213, 219, 0.1);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 1rem;
                ">
                    <span style="font-size: 2rem;">🎯</span>
                    <h3 style="
                        color: #f9fafb;
                        font-size: 1.1rem;
                        font-weight: 600;
                        margin: 0;
                    ">Custom Tokenization</h3>
                </div>
                <p style="
                    color: #9ca3af;
                    font-size: 0.9rem;
                    margin: 0;
                    line-height: 1.5;
                ">Advanced BPE tokenization system designed for efficient text processing and improved understanding.</p>
            </div>
            """)
    
    # Footer Information
    gr.HTML("""
    <div style="
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(209, 213, 219, 0.1);
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
    ">
        <p style="margin: 0;">
            🚀 Master LLM Chat v2.0 | Built with ❤️ using Gradio & PyTorch
        </p>
    </div>
    """)

    def send_message(message, chat_history, max_new_tokens, temperature, top_k, top_p):
        if not message.strip():
            return chat_history, ""
        # Önce typing indicator ekle
        typing_row = [message, "<span class=\"typing-indicator\">Yazıyor <span class=\"dot\"></span><span class=\"dot\"></span><span class=\"dot\"></span></span>"]
        chat_history = chat_history + [typing_row]
        # Gerçek cevabı üret
        reply = generate_reply_text(message, max_new_tokens, temperature, top_k, top_p)
        # Typing indicator'ı gerçek yanıtla değiştir
        chat_history[-1] = [message, reply]
        return chat_history, ""

    # Ek aksiyon fonksiyonları
    def regenerate_last(chat_history, max_new_tokens, temperature, top_k, top_p):
        if not chat_history:
            return chat_history
        last_user = None
        for user, _ in reversed(chat_history):
            if user and isinstance(user, str):
                last_user = user
                break
        if last_user is None:
            return chat_history
        reply = generate_reply_text(last_user, max_new_tokens, temperature, top_k, top_p)
        chat_history[-1][1] = reply
        return chat_history

    def delete_last(chat_history):
        if chat_history:
            chat_history = chat_history[:-1]
        return chat_history

    # Preset mapping for quick settings (radio based)
    def apply_quick_preset(preset_name):
        mapping = {
            "Conservative": (0.3, 20, 0.8),
            "Balanced": (1.0, 50, 0.9),
            "Creative": (1.5, 80, 0.95),
            "Random": (1.8, 100, 0.98),
        }
        return mapping.get(preset_name, (1.0, 50, 0.9))

    # Event handlers
    send_button.click(send_message, [msg, chatbot, max_new_tokens, temperature, top_k, top_p], [chatbot, msg])
    msg.submit(send_message, [msg, chatbot, max_new_tokens, temperature, top_k, top_p], [chatbot, msg])
    clear_button.click(lambda: None, None, chatbot)
    regenerate_button.click(regenerate_last, [chatbot, max_new_tokens, temperature, top_k, top_p], chatbot)
    delete_button.click(delete_last, chatbot, chatbot)
    export_button.click(lambda ch: "\n\n".join([f"User: {u}\nAssistant: {a}" for u, a in ch]) if ch else "", chatbot, None)
    copy_button.click(lambda ch: (ch[-1][1] if ch else ""), chatbot, None)
    reload_button.click(reload_model, inputs=None, outputs=[status])
    load_button.click(load_model_from_url, inputs=[custom_model_url], outputs=[status])
    
    # Quick preset (radio) handler
    quick_preset.change(apply_quick_preset, quick_preset, [temperature, top_k, top_p])

if __name__ == "__main__":
    demo.launch(share=True)