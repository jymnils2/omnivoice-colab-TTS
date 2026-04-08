# %cd /content/omnivoice-colab
import os
import sys
import logging
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import re
import uuid
import traceback 

temp_audio_dir="./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup path to import subtitle_maker from /content/omnivoice-colab/OmniVoice/
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker

# Attempt to import Whisper's supported language dict to filter unsupported languages
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("omnivoice")
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Filtrado de Idiomas
# ---------------------------------------------------------------------------
ALLOWED_LANGS = ["Spanish", "English", "Chinese", "French", "Portuguese"]
_ALL_LANGUAGES = ["Auto"] + ALLOWED_LANGS

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
print("Loading model from k2-fsa/OmniVoice to cuda ...")

from hf_mirror import download_model

try:
  model = OmniVoice.from_pretrained(
      "k2-fsa/OmniVoice",
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
except Exception as e:
  print(f"Error loading from HF, attempting mirror download: {e}")
  omnivoice_model_path=download_model(
    "k2-fsa/OmniVoice",
    download_folder="./OmniVoice_Model",
    redownload=False,
    workers=6,
    use_snapshot=False,
  )

  model = OmniVoice.from_pretrained(
      omnivoice_model_path,
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", 
    "[dissatisfaction-hnn]"
]

INSERT_TAG_JS_VC = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vc_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

INSERT_TAG_JS_VD = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vd_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
}

# ---------------------------------------------------------------------------
# Core Logic & Helpers
# ---------------------------------------------------------------------------
def _is_whisper_supported(lang):
    if not lang or lang == "Auto": return True 
    if WHISPER_LANGUAGE_CODE is None: return True 
    supported_langs = [str(k).lower() for k in WHISPER_LANGUAGE_CODE.keys()] + \
                      [str(v).lower() for v in WHISPER_LANGUAGE_CODE.values()]
    lang_lower = lang.lower()
    for w_lang in supported_langs:
        if w_lang in lang_lower or lang_lower in w_lang: return True
    return False

def generate_subtitles_if_needed(wav_path, lang, want_subs):
    if not want_subs: return None, None, None
    try:
        whisper_lang = lang if (lang and lang != "Auto") else None
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3] 
    except Exception as e:
        print(f"Subtitle error: {e}")
    return None, None, None

def tts_file_name(text, language="en"):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip().replace(" ", "_")
    if not clean_text: clean_text = "audio"
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    rand = uuid.uuid4().hex[:8].upper()
    return f"{temp_audio_dir}/{clean_text[:20]}_{lang}_{rand}.wav"

def _gen_core(
    text, language, ref_audio, instruct, num_step, guidance_scale, 
    denoise, speed, duration, preprocess_prompt, postprocess_output, mode, ref_text=None
):
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    try:
        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None
        kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

        if speed is not None and float(speed) != 1.0: kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0: kw["duration"] = float(duration)

        # Lógica mejorada: Solo usa clonación si hay un audio de referencia
        if mode == "clone" and ref_audio:
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
        
        if mode == "design" and instruct:
            kw["instruct"] = instruct.strip()

        audio = model.generate(**kw)
        waveform = audio[0].squeeze(0).numpy()
        waveform = (waveform * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    except Exception as e:
        print(f"Synthesis Error: {e}")
        traceback.print_exc()
        return None, f"Error: {e}"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = ".tag-container { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; } .tag-btn { min-width: fit-content; height: 32px; font-size: 13px; background: #eef2ff; border: 1px solid #c7d2fe; color: #3730a3; border-radius: 6px; cursor: pointer; }"

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(label=label, choices=_ALL_LANGUAGES, value=value, interactive=True)

def _gen_settings():
    with gr.Accordion("Settings", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
        du = gr.Number(value=None, label="Duration (s)")
        ns = gr.Slider(4, 64, value=32, step=1, label="Steps")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG Scale")
        pp = gr.Checkbox(label="Preprocess", value=True)
        po = gr.Checkbox(label="Postprocess", value=True)
    return ns, gs, dn, sp, du, pp, po

with gr.Blocks(theme=theme, css=css) as demo:
    gr.HTML("<h1 style='text-align: center;'>🎙️ OmniVoice Multilingual</h1>")

    with gr.Tabs():
        with gr.TabItem("Voice Clone / TTS"):
            with gr.Row():
                with gr.Column():
                    vc_text = gr.Textbox(label="Text", lines=4, elem_id="vc_textbox")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(None, [btn, vc_text], vc_text, js=INSERT_TAG_JS_VC)
                    with gr.Row():
                        vc_lang = _lang_dropdown()
                        vc_want_subs = gr.Checkbox(label="Subtitles?", value=False)
                    vc_ref_audio = gr.Audio(label="Reference Audio (Optional for TTS)", type="filepath")
                    vc_ref_text = gr.Textbox(label="Reference Text (Optional)", lines=2)
                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                with gr.Column():
                    vc_audio = gr.Audio(label="Output")
                    vc_status = gr.Textbox(label="Status")
                    vc_out_wav = gr.File(label="Download WAV")

            def _auto_transcribe(audio_path, lang):
                if not audio_path: return ""
                try:
                    whisper_results = subtitle_maker(audio_path, lang if lang != "Auto" else None)
                    return whisper_results[7] if whisper_results and len(whisper_results) > 7 else ""
                except: return ""

            vc_ref_audio.change(_auto_transcribe, [vc_ref_audio, vc_lang], vc_ref_text)

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                # Si no hay audio, el modo pasa internamente a TTS estándar
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
                if res[0] is None: return None, res[1], None
                audio_tuple, status = res
                tmp_wav = tts_file_name(text, lang)
                wavfile.write(tmp_wav, audio_tuple[0], audio_tuple[1])
                return audio_tuple, status, tmp_wav

            vc_btn.click(_clone_fn, [vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po], [vc_audio, vc_status, vc_out_wav])

        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column():
                    vd_text = gr.Textbox(label="Text", lines=4, elem_id="vd_textbox")
                    vd_lang = _lang_dropdown()
                    vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Design", open=False):
                        vd_gender = gr.Dropdown(label="Gender", choices=["Auto", "Male", "Female"], value="Female")
                        vd_age = gr.Dropdown(label="Age", choices=["Auto", "Child", "Young Adult", "Elderly"], value="Young Adult")
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                with gr.Column():
                    vd_audio = gr.Audio(label="Output")
                    vd_status = gr.Textbox(label="Status")

            def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, gender, age):
                instruct = f"{gender}, {age}"
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
                return res[0], res[1]

            vd_btn.click(_design_fn, [vd_text, vd_lang, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po, vd_gender, vd_age], [vd_audio, vd_status])

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)