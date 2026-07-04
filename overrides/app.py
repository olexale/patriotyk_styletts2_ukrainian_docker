import glob
import os
import re
import gradio as gr

import spaces
from verbalizer import Verbalizer

import torch
from ipa_uk import ipa
from unicodedata import normalize
from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier, StressSymbol
stressify = Stressifier()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompts_dir = 'voices'

verbalizer = Verbalizer()


def split_to_parts(text, group=True):
    text = re.sub(r'([^.,!:?\-])\n', r'\1. ', text)
    text = text.replace('\n', ' ')
    split_symbols = '.?!:'
    parts = ['']
    index = 0
    last = len(text)-1
    for i, s in enumerate(text):
        parts[index] += s
        if s in split_symbols and i < last and text[i+1] == ' ':
            if group and len(parts[index]) <= 20:
                continue
            index += 1
            parts.append('')
    return parts



single_model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_single', device=device)
single_style = torch.load('filatov.pt')


multi_model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_multispeaker', device=device)
multi_styles = {}

prompts_list = sorted(glob.glob(os.path.join(prompts_dir, '*.pt')))
prompts_list = ['.'.join(p.split('/')[-1].split('.')[:-1]) for p in prompts_list]

for audio_prompt in prompts_list:
    audio_path = os.path.join(prompts_dir, audio_prompt+'.pt')
    multi_styles[audio_prompt] = torch.load(audio_path)
    print('loaded ', audio_prompt)

models = {
    'multi': {
        'model': multi_model,
        'styles': multi_styles
    },
    'single': {
        'model': single_model,
        'style': single_style
    }
}


def verbalize(text):
    parts = split_to_parts(text, group=False)
    verbalized = ''
    for part in parts:
        if part.strip():
            verbalized += verbalizer.process_text(part.strip())[0] + ' '
    return verbalized

description = f'''
<h1 style="text-align:center;">StyleTTS2 ukrainian demo</h1><br>
Програма може не коректно визначати деякі наголоси.
Якщо наголос не правильний, використовуйте символ + після наголошеного складу.
Текст який складається з одного слова може синтезуватися з певними артефактами, пишіть повноцінні речення.
Якщо текст містить цифри чи акроніми, можна натисну кнопку "Вербалізувати" яка повинна замінити цифри і акроніми
в словесну форму.

'''

examples = [
    ["Решта окупантів звернула на Вокзальну — центральну вулицю Бучі. Тільки уявіть їхній настрій, коли перед ними відкрилася ця пасторальна картина! Невеличкі котеджі й просторіші будинки шикуються обабіч, перед ними вивищуються голі липи та електро-стовпи, тягнуться газони й жовто-чорні бордюри. Доглянуті сади визирають із-поза зелених парканів, гавкотять собаки, співають птахи… На дверях будинку номер тридцять шість досі висить різдвяний вінок.", 1.0],
    ["Одна дівчинка стала королевою Франції. Звали її Анна, і була вона донькою Ярослава Му+дрого, великого київського князя. Він опі+кувався літературою та культурою в Київській Русі+, а тоді переважно про таке не дбали – більше воювали і споруджували фортеці.", 1.0],
    ["Одна дівчинка народилася і виросла в Америці, та коли стала дорослою, зрозуміла, що дуже любить українські вірші й найбільше хоче робити вистави про Україну. Звали її Вірляна. Дід Вірляни був український мовознавець і педагог Кость Кисілевський, котрий навчався в Лейпцизькому та Віденському університетах і, після Другої світової війни виїхавши до США, започаткував систему шкіл українознавства по всій Америці. Тож Вірляна зростала в українському середовищі, а окрім того – в середовищі вихідців з інших країн.", 1.0],
    ["За інформацією від Державної служби з надзвичайних ситуацій станом на 7 ранку 15 липня.", 1.0],
    ["Очікується, що цей застосунок буде запущено 22.08.2025.", 1.0],
]



def _synth_core(model_name, text, speed, voice_name=None):
    """Core synthesis shared by the audio-only and timed endpoints.

    Returns (sample_rate, audio_numpy, visemes) where visemes is a flat list of
    {phoneme, start_ms, dur_ms} spanning the WHOLE utterance — each part's local
    timeline is shifted by the cumulative audio duration of the preceding parts.
    """
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 50000:
        raise gr.Error("Text must be <50k characters")

    model = models[model_name]['model']
    result_wav = []
    visemes = []
    offset_ms = 0.0
    for t in split_to_parts(text):
        t = t.strip()
        t = t.replace('"', '')
        if not t:
            continue
        t = t.replace('+', StressSymbol.CombiningAcuteAccent)
        t = normalize('NFKC', t)
        t = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', t)
        if t[-1] not in '.?!:-':
            t += '.'
        t = re.sub(r' - ', ': ', t)
        t = stressify(t)
        ps = ipa(t)
        if not ps:
            continue

        if voice_name:
            style = models[model_name]['styles'][voice_name]
        else:
            style = models[model_name]['style']

        wav, part_visemes = model.synthesize_aligned(ps, speed=speed, s_prev=style)
        for v in part_visemes:
            visemes.append({**v, 'start_ms': round(v['start_ms'] + offset_ms, 2)})
        result_wav.append(wav)
        offset_ms += wav.shape[-1] / 24000 * 1000.0

    audio = torch.concatenate(result_wav).cpu().numpy()
    return 24000, audio, visemes


@spaces.GPU
def synthesize(model_name, text, speed, voice_name = None, progress=gr.Progress()):
    print("*** saying ***")
    print(text)
    print("*** end ***")
    sr, audio, _ = _synth_core(model_name, text, speed, voice_name)
    return sr, audio


@spaces.GPU
def synthesize_timed(model_name, text, speed, voice_name=None):
    """API-only: same audio as /synthesize plus the phoneme viseme timeline."""
    sr, audio, visemes = _synth_core(model_name, text, speed, voice_name)
    return (sr, audio), visemes



def select_example(df, evt: gr.SelectData):
    return evt.row_value   
    
with gr.Blocks() as single:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            verbalize_button = gr.Button("Вербалізувати(beta)")
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            verbalize_button.click(verbalize, inputs=[input_text], outputs=[input_text])
            
        with gr.Column(scale=1):
            output_audio = gr.Audio(
                    label="Audio:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
            synthesise_button = gr.Button("Синтезувати")
            single_text = gr.Text(value='single', visible=False)
            synthesise_button.click(synthesize, inputs=[single_text, input_text, speed], outputs=[output_audio])
    
    with gr.Row():
        examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False)
        examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])
    
with gr.Blocks() as multy:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            verbalize_button = gr.Button("Вербалізувати(beta)")
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            speaker = gr.Dropdown(label="Голос:", choices=prompts_list, value=prompts_list[0])
            verbalize_button.click(verbalize, inputs=[input_text], outputs=[input_text])

        with gr.Column(scale=1):
            output_audio = gr.Audio(
                    label="Audio:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
            synthesise_button = gr.Button("Синтезувати")
            multi = gr.Text(value='multi', visible=False)
            
            synthesise_button.click(synthesize, inputs=[multi, input_text, speed, speaker], outputs=[output_audio])
    with gr.Row():
        examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False, show_label=True, label="Приклади:")
        examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])




with gr.Blocks(title="StyleTTS2 ukrainian demo", css="") as demo:
    gr.Markdown(description)
    gr.TabbedInterface([multy, single], ['Multі speaker', 'Single speaker'])

    # API-only endpoint: audio + phoneme viseme timeline. Not shown in the UI;
    # reachable via gradio_client predict(api_name="/synthesize_timed").
    api_model = gr.Text(visible=False)
    api_text = gr.Text(visible=False)
    api_speed = gr.Number(value=1.0, visible=False)
    api_voice = gr.Text(visible=False)
    api_audio_out = gr.Audio(type="numpy", visible=False)
    api_visemes_out = gr.JSON(visible=False)
    api_btn = gr.Button(visible=False)
    api_btn.click(
        synthesize_timed,
        inputs=[api_model, api_text, api_speed, api_voice],
        outputs=[api_audio_out, api_visemes_out],
        api_name="synthesize_timed",
    )


if __name__ == "__main__":
    demo.queue(api_open=True, max_size=15).launch(show_api=True)