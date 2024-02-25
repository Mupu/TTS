import os
import sys
import logging
import gradio as gr
import torch
import traceback
from pathlib import Path
from utils.formatter import format_audio_list
from utils.gpt_train import train_gpt

#  ------------------
# define a logger to redirect 
class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

#  ------------------
def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()

#  ------------------
def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

#  ------------------
def preprocess_dataset(audio_path, language, whisper_model, out_path, progress=gr.Progress(track_tqdm=True)):
    clear_gpu_cache()
    # todo modelname
    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)
    if audio_path is None:
        return "You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!", "", ""
    else:
        try:
            train_meta, eval_meta, audio_total_size = format_audio_list(audio_path, target_language=language, whisper_model=whisper_model, out_path=out_path, gradio_progress=progress)
        except:
            traceback.print_exc()
            error = traceback.format_exc()
            return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""

    clear_gpu_cache()

    # if audio total len is less than 2 minutes raise an error
    if audio_total_size < 120:
        message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
        print(message)
        return message, "", ""

    print("Dataset Processed!")
    return "Dataset Processed!", train_meta, eval_meta

#  ------------------
def load_params(model_name):
    # todo
    path_output = Path("./finetuned_models") / model_name

    dataset_path = path_output / "dataset"

    if not dataset_path.exists():
        return "The output folder does not exist!", "", "", "en"

    eval_train = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"

    # Write the target language to lang.txt in the output directory
    lang_file_path = dataset_path / "lang.txt"

    # Check if lang.txt already exists and contains a different language
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    clear_gpu_cache()

    print(current_language)
    return "The data has been updated", eval_train, eval_csv, current_language

#  ------------------
def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()
    if not train_csv or not eval_csv:
        return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
    try:
        # convert seconds to waveform frames
        max_audio_length = int(max_audio_length * 22050)
        config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
    except:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

    # copy original files to avoid parameters changes issues
    os.system(f"cp {config_path} {exp_path}")
    os.system(f"cp {vocab_file} {exp_path}")

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("Model training done!")
    clear_gpu_cache()
    return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav




# redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    # ---------------
    # ui
    with gr.Blocks() as trainer:
        with gr.Tab("1 - Data processing & Training"):
            gr.Markdown("# 1 Stage - Prepare Dataset")
            load_params_btn = gr.Button(value="Load Params from output folder")
            custom_model_name = gr.Textbox(
                label="Finetune Model Name",
                value="my_finetune_model",
                interactive=True,
            )
            out_path = gr.Textbox(
                label="Output path (where data and checkpoints will be saved):",
            )
            gr.Label("File Source:")
            with gr.Tab("GDrive"):
                dataset_path = gr.Textbox(
                    label="Dataset path (where the audio files are):",
                )

            with gr.Tab("Upload Files"):
                upload_file = gr.File(
                    file_count="multiple",
                    label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
                )

            lang = gr.Dropdown(
                label="Dataset Language",
                value="de",
                choices=[ "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja"],
            )

            train_whisper_model = gr.Dropdown(
                label="Whisper Model",
                value="large-v3",
                choices=["large-v3", "large-v2", "large", "medium", "small"],
            )


            ## training
            gr.Markdown("# 2 Stage - Finetune XTTS")
            train_csv = gr.Textbox(
                label="Train CSV:",
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
            )
            num_epochs =  gr.Slider(
                label="Number of epochs:",
                minimum=1,
                maximum=100,
                step=1,
                value=10,
            )
            batch_size = gr.Slider(
                label="Batch size:",
                minimum=2,
                maximum=512,
                step=1,
                value=10,
            )
            grad_acumm = gr.Slider(
                label="Grad accumulation steps:",
                minimum=2,
                maximum=128,
                step=1,
                value=2,
            )
            max_audio_length = gr.Slider(
                label="Max permitted audio size in seconds:",
                minimum=2,
                maximum=20,
                step=1,
                value=20,
            )
            train_version = gr.Dropdown(
                label="XTTS base version",
                value="v2.0.3",
                choices=["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"],
            )
            progress_train = gr.Label(
                label="Progress:"
            )
            logs_tts_train = gr.Textbox(
                label="Logs:",
                interactive=False,
            )
            trainer.load(read_logs, None, logs_tts_train, every=1)

            train_status_bar = gr.Label(
                label="Train Status Bar", value="Load data, choose options and click Train")

            train_btn = gr.Button(value="Step 2 - Run the training")

    

        # ---------------
        # functions
        load_params_btn.click(fn=load_params, inputs=[custom_model_name], outputs=[
                train_status_bar, train_csv, eval_csv, lang])

    # ---------------
    # launch
    trainer.launch(
        inbrowser=True, 
        share=True,
        debug=False,
        server_port=666,
        server_name="0.0.0.0"
    )