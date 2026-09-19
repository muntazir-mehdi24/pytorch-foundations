import os
import shutil

# mapping = {"JPEG" : "IMAGES", "PNG" : "IMAGES", "JPG" : "IMAGES", "GIF" : "IMAGES", "TXT" : "DOCUMENTS", "PDF" : "DOCUMENTS", "CSV" : "CSV FILES", "PY" : "PYTHON FILES", "MP3" : "AUDIO FILES", "MP4" : "VIDEO FILES", "MKV" : "VIDEO FILES", "EXE" : "EXECUTABLES", "ZIP" : "ARCHIVES", "RAR" : "ARCHIVES", "DOCX" : "DOCUMENTS", "PPTX" : "DOCUMENTS", "XLSX" : "SHEETS"}

target_dir = os.path.expanduser("~/Downloads")

for file in os.listdir(target_dir):
    file_path = os.path.join(target_dir, file)
    if os.path.isfile(file_path):

    # extract the file extension
        file_ext = os.path.splitext(file)[1][1:].upper()

        # check if the file extension is in the dictionary
        if file_ext in mapping:
            folder_name = mapping[file_ext]
            folder_path = os.path.join(target_dir, folder_name)

            os.makedirs(folder_path, exist_ok=True)

            # move the file to the corresponding folder
            shutil.move(file_path, os.path.join(folder_path, file))