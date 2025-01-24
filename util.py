import streamlit as st
import os

def does_file_have_pdf_extension(file):
    if not (file.name.endswith('.pdf')):
        st.warning('Not a valid PDF file.', icon="⚠️")
        return False
    return True

# def store_pdf_file(file, dir):
#     file_path = os.path.join(dir, file.name)
#     with open(file_path, 'wb') as fhand:
#         fhand.write(file.getbuffer())
#     return file_path

def store_pdf_file(file, directory):
    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path
