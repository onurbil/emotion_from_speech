from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg

import scipy
import librosa
from dataset import calculate_features

import os
import shutil
from pydub import AudioSegment as am

AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function toggleRecording() {
  recorder.stop();
  gumStream.getAudioTracks()[0].stop();
  recordButton.innerText = "Saving the recording... pls wait!"
}

var handleSuccess = function(stream) {
  gumStream = stream;
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {            
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data); 
    reader.onloadend = function() {
      base64data = reader.result;
    }
  };
};
navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);
sleep(1000).then(() =>{
  recordButton.appendChild(t)
});

var data = new Promise(resolve=>{
  recordButton.onclick = ()=>{
    recordButton.innerText = "Recording for 2 seconds"
    recorder.start()
    sleep(3000).then(() =>{
      toggleRecording()
      sleep(2000).then(() => {
        resolve(base64data.toString())
      });
    });
  }
});
</script>
"""
import time


def get_audio():
    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])

    process = (ffmpeg
               .input('pipe:0')
               .output('pipe:1', format='wav')
               .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
               )
    output, err = process.communicate(input=binary)

    riff_chunk_size = len(output) - 8
    # Break up the chunk size into four bytes, held in b.
    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)

    # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
    riff = output[:4] + bytes(b) + output[8:]

    sr, audio = wav_read(io.BytesIO(riff))

    return audio, sr

def save_audio(DATA_PATH, name = 'recording'):
    data_list = []
    cls = 'ANY'
    ## Say the word calm
    wav = name + '.wav'
    npy = name + '.npy'

    # Get Audio
    audio, sr = get_audio()

    scipy.io.wavfile.write(wav, sr, audio)  # Saving audio
    sound = am.from_file(wav, format='wav', frame_rate=48000, start_second=0.2, duration=1.9)
    sound = sound.set_frame_rate(24414)
    sound.export(wav, format='wav')
    data, sr = librosa.load(wav, sr=24414)
    features = calculate_features(data, sr, n_mfcc=20, order=1)
    data_list.append([features, cls])
    data_list = np.array(data_list, dtype=object)
    np.save(npy, data_list)

    filesToMove = [
                    wav,
                    npy
                  ]

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    for files in filesToMove:
        shutil.copy(files, DATA_PATH)