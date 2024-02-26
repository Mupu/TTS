import os
import sys
from pathlib import Path


path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
#print(sys.path)

import logging
import gradio as gr
import torch
import traceback
import shutil
from utils.formatter import format_audio_list, find_latest_best_model
from utils.gpt_train import train_gpt
#todo properly continue training
#todo loss graph

#todo save every x epochs
#todo early training stop
#todo check why old ui has 44100kh and not 22050


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
def get_all_files_in_directory(directory):
        # Initialize an empty list to store full path names
        file_paths = []
        
        print(directory)

        # Walk through the directory and its subdirectories
        for root, directories, files in os.walk(directory):
            # Iterate over each file in the current directory
            for f in files:
                # Construct the full path of the file and append it to the list
                file_path = os.path.join(root, f)
                file_paths.append(file_path)

        return file_paths

#  ------------------
def load_params(root_path, model_name):
    root_path = Path(root_path)
    if not root_path.exists():
        gr.Info("Could not find and load params.")
        print("Could not find and load params.")
        return "", "", "de"

    path_output = root_path / model_name / "training"

    # this is the processed dataset by whisper, not the original the user supplied
    dataset_path = path_output / "dataset"

    if not dataset_path.exists():
        model_names = [folder for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]
        # only warn if its not a new model
        if model_name in model_names:
            gr.Info("Could not find and load params.")
            print("Could not find and load params.")
        return "", "", "de"

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

    return eval_train, eval_csv, current_language

#  ------------------
def preprocess_dataset(audio_path, language, whisper_model, training_path, status_bar):
    clear_gpu_cache()
    training_path = Path(training_path) / "dataset"
    os.makedirs(training_path, exist_ok=True)
    
    train_meta = "" 
    eval_meta = ""
    audio_total_size = 0;

    if audio_path is None:
        msg = "You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!"
        print(msg)
        gr.Error(msg)
        return "", ""
    else:
        try:
            train_meta, eval_meta, audio_total_size = format_audio_list(audio_path, target_language=language, whisper_model=whisper_model, out_path=training_path, gradio_progress=status_bar)
        except:
            traceback.print_exc()
            error = traceback.format_exc()
            return "", ""

    clear_gpu_cache()

    # if audio total len is less than 2 minutes raise an error
    if audio_total_size < 120:
        message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
        print(message)
        gr.Error(message)
        return "", ""

    print("Dataset Processed!")
    gr.Info("Dataset Processed!")
    return train_meta, eval_meta

#  ------------------
def optimize_model(training_path, clear_train_data):
    training_path = Path(training_path)  # Ensure that training_path is a Path object.

    ready_dir = training_path / "ready"
    run_dir = training_path / "run"
    dataset_dir = training_path / "dataset"

    # Clear specified training data directories.
    if clear_train_data in {"run", "all"} and run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {run_dir}: {e}")

    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        try:
            shutil.rmtree(dataset_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {dataset_dir}: {e}")

    # Get full path to model
    model_path = ready_dir / "unoptimize_model.pth"

    if not model_path.is_file():
        return "Unoptimized model not found in ready folder", ""

    # Load the checkpoint and remove unnecessary parts.
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    os.remove(model_path)

    # Save the optimized model.
    optimized_model_file_name = "model.pth"
    optimized_model = ready_dir/optimized_model_file_name

    torch.save(checkpoint, optimized_model)
    ft_xtts_checkpoint = str(optimized_model)

    clear_gpu_cache()

    return f"Model optimized and saved at {ft_xtts_checkpoint}!", ft_xtts_checkpoint

#  ------------------
def train_model(model_to_train, xtts_version, language, num_epochs, batch_size, grad_acumm, 
                root_path, model_name, whisper_model, max_audio_length, 
                dataset_active_tab, dataset_local_folder_path, dataset_uploaded_files):
    clear_gpu_cache()
    status_bar = None

    audio_data = None
    if dataset_active_tab == "local_folder":
        audio_data = get_all_files_in_directory(dataset_local_folder_path)
    elif dataset_active_tab == "file_upload":
        audio_data = dataset_uploaded_files
    else:
        print("Should never happen, no audio data")
        gr.Error("Should never happen, no audio data")
        return 

    msg = None
    if audio_data is None or len(audio_data) == 0:
        msg = "audio data can not be empty"
    if model_name is None or (type(model_name) == str and len(model_name.strip()) == 0)\
        or (type(model_name) == list and len(model_name) == 0):
        msg = "model name can not be empty"
    if root_path is None or len(root_path.strip()) == 0:
        msg = "root path can not be empty"
        
    if msg is not None:
        print(msg)
        gr.Info(msg)
        return


    training_path = Path(root_path) / model_name / "training"
    training_path.mkdir(parents=True, exist_ok=True)

    train_csv, eval_csv = preprocess_dataset(audio_data, language, whisper_model, training_path, status_bar)

    # Check if the dataset language matches the language you specified
    lang_file_path = training_path / "dataset" / "lang.txt"

    # Check if lang.txt already exists and contains a different language
    current_language = None
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
            if current_language != language:
                print("The language that was prepared for the dataset does not match the specified language. Change the language to the one specified in the dataset")
                gr.Error("The language that was prepared for the dataset does not match the specified language. Change the language to the one specified in the dataset")
                language = current_language

    if not train_csv or not eval_csv:
        print("Should never happen, no train_csv or eval_csv")
        gr.Error("Should never happen, no train_csv or eval_csv")
        return 
    try:
        # convert seconds to waveform frames
        max_audio_length = int(max_audio_length * 22050)
        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            model_to_train, xtts_version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=training_path, max_audio_length=max_audio_length)
        print("oricheck: " + original_xtts_checkpoint)
    except:
        traceback.print_exc()
        error = traceback.format_exc()
        return

    # copy original files to avoid parameters changes issues
    # os.system(f"cp {config_path} {exp_path}")
    # os.system(f"cp {vocab_file} {exp_path}")

    ready_dir = Path(training_path) / "ready"

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("ft_xtts_check: " + ft_xtts_checkpoint)

    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
    # os.remove(ft_xtts_checkpoint)

    ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")
    print("ft_xtts_check2: " + ft_xtts_checkpoint)

    # Reference
    # Move reference audio to output folder and rename it
    speaker_reference_path = Path(speaker_wav)
    print(f"speaker wav: {speaker_reference_path}")
    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_reference_path, speaker_reference_new_path)

    optimize_model(training_path, "none")

    print("Model training done! Model moved into models folder")
    gr.Info("Model training done! Model moved into models folder")

    # clear_gpu_cache()
    return



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
            root_path = gr.Textbox(
                label="root path (where all the model folders are located):",
                interactive=True,
            )

            model_name = gr.Dropdown(
                label="(New) Model Name",
                allow_custom_value=True,
                interactive=True,
            )
            

            dataset_active_tab = gr.State("local_folder")
            with gr.Tab("Audio Dataset") as dataset_tabs:
                with gr.Tab("Local Folder", id="local_folder") as dataset_local_folder_path_tab:
                    dataset_local_folder_path = gr.Textbox(
                        label="Dataset path (where the audio files are):",
                        value="/content/drive/",
                        interactive=True,
                    )

                with gr.Tab("Upload Files", id="file_upload") as dataset_uploaded_files_tab:
                    dataset_uploaded_files = gr.File(
                        file_count="multiple",
                        label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
                    )

            lang = gr.Dropdown(
                label="Dataset Language",
                value="de",
                choices=[ "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja"],
                interactive=True,
            )

            whisper_model = gr.Dropdown(
                label="Whisper Model",
                value="large-v3",
                choices=["large-v3", "large-v2", "large", "medium", "small"],
                interactive=True,
            )


            ## training
            gr.Markdown("# 2 Stage - Finetune XTTS")
            model_to_train = gr.Textbox(
                label="Model to train. Leave empty to start from base model",
                interactive=True,
            )
            xtts_version = gr.Dropdown(
                label="XTTS base version",
                value="v2.0.3",
                choices=["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"],
                interactive=True,
            )
            train_csv = gr.Textbox(
                label="Train CSV:",
                interactive=False,
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
                interactive=False,
            )
            num_epochs =  gr.Slider(
                label="Number of epochs:",
                minimum=1,
                maximum=100,
                step=1,
                value=10,
                interactive=True,
            )
            batch_size = gr.Slider(
                label="Batch size:",
                minimum=1,
                maximum=512,
                step=1,
                value=10,
                interactive=True,
            )
            grad_acumm = gr.Slider(
                label="Grad accumulation steps:",
                minimum=1,
                maximum=128,
                step=1,
                value=2,
                interactive=True,
            )
            max_audio_length = gr.Slider(
                label="Max permitted audio size in seconds:",
                minimum=2,
                maximum=20,
                step=1,
                value=20,
                interactive=True,
            )
            logs_tts_train = gr.Textbox(
                label="Logs:",
                interactive=False,
            )
            trainer.load(read_logs, None, logs_tts_train, every=1)

            train_btn = gr.Button(value="Start Training")

    

        # ---------------
        # functions
        # Callback function to update dropdown options when root path changes
        def update_dropdown_options(root_path):
            try:
                if os.path.isdir(root_path):
                        folder_names = [folder for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]
                        if folder_names:
                            first_folder = folder_names[0]
                            return gr.Dropdown(choices=folder_names, value=first_folder)
                        else:
                            return gr.Dropdown(choices=[], value=None)
                else:
                    return gr.Dropdown(choices=[], value=None)
            except:
                return gr.Dropdown(choices=[], value=None)
        root_path.change(fn=update_dropdown_options, inputs=root_path, outputs=model_name)

        # ---------------
        def update_dataset_path(root_path, model_name):
            if model_name is not None and len(model_name.strip()) > 0\
                and root_path is not None and len(root_path.strip()) > 0:

                model_to_train = find_latest_best_model(str(Path(root_path) / model_name))
                if model_to_train is None:
                    model_to_train = ""
                return gr.Textbox(Path(root_path) / model_name / "dataset"), *load_params(root_path, model_name), model_to_train
            else:
                return "Choose a model first, or create a new one", "", "", "de", "none"
        model_name.change(fn=update_dataset_path, inputs=[root_path, model_name], 
                          outputs=[dataset_local_folder_path, train_csv, eval_csv, lang, model_to_train])

        # ---------------

        def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData
            return evt.target.id
        dataset_local_folder_path_tab.select(fn=on_select, outputs=dataset_active_tab )
        dataset_uploaded_files_tab.select(fn=on_select, outputs=dataset_active_tab )

        # ---------------
        train_btn.click(fn=train_model, inputs=[model_to_train, xtts_version, lang, num_epochs,
                                                batch_size, grad_acumm, root_path, model_name, 
                                                whisper_model, max_audio_length,
                                                dataset_active_tab, dataset_local_folder_path, dataset_uploaded_files]
                                                , outputs=[])

    # ---------------
    # launch
    trainer.launch(
        inbrowser=True, 
        share=True,
        debug=False,
        server_port=6666,
        server_name="0.0.0.0"
    )