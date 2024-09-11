import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import threading
import os
from datetime import datetime

recognition_process = None

# Function to get the absolute path of scripts
def get_script_path(script_name):
    return os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', script_name)

def run_capture_images():
    def target():
        name = entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return
        script_path = get_script_path('capture_images.py')
        try:
            subprocess.run(["python", script_path, name], check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to capture images: {e}")

    threading.Thread(target=target).start()

def run_train_model():
    def target():
        script_path = get_script_path('train_model.py')
        try:
            subprocess.run(["python", script_path], check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    threading.Thread(target=target).start()

def run_recognize_face():
    global recognition_process
    def target():
        script_path = get_script_path('recognize_face.py')
        try:
            global recognition_process
            recognition_process = subprocess.Popen(["python", script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start face recognition: {e}")

    threading.Thread(target=target).start()

def stop_recognize_face():
    global recognition_process
    if recognition_process:
        recognition_process.terminate()
        recognition_process.wait()  # Wait for the process to terminate
        recognition_process = None
        messagebox.showinfo("Info", "Face recognition stopped.")
    else:
        messagebox.showerror("Error", "Face recognition is not running.")

def shutdown():
    root.destroy()

def search_documents():
    year = year_combobox.get()
    month = month_combobox.get()
    day = day_combobox.get()
    file_name = file_name_entry.get()
    
    if not (year and month and day and file_name):
        messagebox.showerror("Error", "Please select year, month, date, and enter a file name.")
        return

    # Construct the filename based on the selected date and entered file name
    filename = f"{file_name}_{year}-{month}-{day}.xlsx"
    document_path = os.path.join(os.path.dirname(__file__), '..', '..', 'records', filename)

    if os.path.exists(document_path):
        # Open the document in Excel
        try:
            os.startfile(document_path)
            messagebox.showinfo("Search Result", f"Opening document: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open document: {str(e)}")
    else:
        messagebox.showerror("Error", f"Document for {year}-{month}-{day} with file name {file_name} not found.")

# GUI Setup
root = tk.Tk()
root.title("Face Recognition System")

# Set Sci-Fi Style
root.configure(bg='#0a0a0a')
root.geometry('400x500')

# Sci-Fi Fonts and Colors
font_large = ("Courier", 14, "bold")
font_medium = ("Courier", 12)
font_small = ("Courier", 10)
color_bg = '#0a0a0a'
color_fg = '#00ff00'
color_btn_bg = '#333333'
color_btn_fg = '#00ff00'

frame = tk.Frame(root, bg=color_bg)
frame.pack(padx=20, pady=20)

label = tk.Label(frame, text="Enter your name:", font=font_medium, bg=color_bg, fg=color_fg)
label.grid(row=0, column=0, padx=5, pady=5)

entry = tk.Entry(frame, font=font_medium, bg=color_btn_bg, fg=color_fg, insertbackground=color_fg)
entry.grid(row=0, column=1, padx=5, pady=5)

capture_button = tk.Button(frame, text="Capture Images", font=font_medium, bg=color_btn_bg, fg=color_fg, command=run_capture_images)
capture_button.grid(row=1, column=0, columnspan=2, pady=5)

train_button = tk.Button(frame, text="Train Model", font=font_medium, bg=color_btn_bg, fg=color_fg, command=run_train_model)
train_button.grid(row=2, column=0, columnspan=2, pady=5)

recognize_button = tk.Button(frame, text="Recognize Face", font=font_medium, bg=color_btn_bg, fg=color_fg, command=run_recognize_face)
recognize_button.grid(row=3, column=0, columnspan=2, pady=5)

stop_recognize_button = tk.Button(frame, text="Stop Recognition", font=font_medium, bg=color_btn_bg, fg=color_fg, command=stop_recognize_face)
stop_recognize_button.grid(row=4, column=0, columnspan=2, pady=5)

# Add Sci-Fi Elements
sci_fi_label = tk.Label(frame, text="Face Recognition System", font=font_large, bg=color_bg, fg=color_fg)
sci_fi_label.grid(row=5, column=0, columnspan=2, pady=10)

# Add Off Button
off_button = tk.Button(frame, text="Off", font=font_medium, bg=color_btn_bg, fg=color_fg, command=shutdown)
off_button.grid(row=6, column=0, columnspan=2, pady=5)

# Add Search Box for Documents
search_frame = tk.Frame(root, bg=color_bg)
search_frame.pack(padx=20, pady=20)

search_label = tk.Label(search_frame, text="Search Documents", font=font_large, bg=color_bg, fg=color_fg)
search_label.grid(row=0, column=0, columnspan=3, pady=10)

# Generate years, months, and days
years = [str(year) for year in range(2000, datetime.now().year + 1)]
months = [str(month).zfill(2) for month in range(1, 13)]
days = [str(day).zfill(2) for day in range(1, 32)]

year_combobox = ttk.Combobox(search_frame, values=years, font=font_medium)
year_combobox.grid(row=1, column=0, padx=5, pady=5)
year_combobox.set("Year")

month_combobox = ttk.Combobox(search_frame, values=months, font=font_medium)
month_combobox.grid(row=1, column=1, padx=5, pady=5)
month_combobox.set("Month")

day_combobox = ttk.Combobox(search_frame, values=days, font=font_medium)
day_combobox.grid(row=1, column=2, padx=5, pady=5)
day_combobox.set("Day")

file_name_label = tk.Label(search_frame, text="File Name:", font=font_medium, bg=color_bg, fg=color_fg)
file_name_label.grid(row=2, column=0, padx=5, pady=5)

file_name_entry = tk.Entry(search_frame, font=font_medium, bg=color_btn_bg, fg=color_fg, insertbackground=color_fg)
file_name_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

search_button = tk.Button(search_frame, text="Search", font=font_medium, bg=color_btn_bg, fg=color_fg, command=search_documents)
search_button.grid(row=3, column=0, columnspan=3, pady=5)

root.mainloop()
