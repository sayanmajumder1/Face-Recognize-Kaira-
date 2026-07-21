import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import threading
import os
import sys
from datetime import datetime

recognition_process = None

# ============================================================================
# PATH CONFIGURATION - FIXED
# ============================================================================

# Get the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up 2 levels: ui/desktop -> ui -> D:\Face-Recognize-Kaira-\
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Scripts directory
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
RECORDS_DIR = os.path.join(PROJECT_ROOT, 'records')

# ============================================================================
# PATH UTILITIES
# ============================================================================

def get_script_path(script_name):
    """Get absolute path to a script in the scripts folder"""
    return os.path.join(SCRIPTS_DIR, script_name)

def get_records_path(filename):
    """Get absolute path to a record file"""
    return os.path.join(RECORDS_DIR, filename)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def run_capture_images():
    def target():
        name = entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return
        
        script_path = get_script_path('capture_images.py')
        
        # Check if script exists
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return
        
        try:
            subprocess.run([sys.executable, script_path, name], check=True)
            messagebox.showinfo("Success", f"✅ Captured images for {name}!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to capture images: {e}")

    threading.Thread(target=target, daemon=True).start()

def run_train_model():
    def target():
        script_path = get_script_path('train_model.py')
        
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return
        
        try:
            subprocess.run([sys.executable, script_path], check=True)
            messagebox.showinfo("Success", "✅ Model trained successfully!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    threading.Thread(target=target, daemon=True).start()

def run_recognize_face():
    global recognition_process
    
    def target():
        script_path = get_script_path('recognize_face.py')
        
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return
        
        try:
            global recognition_process
            recognition_process = subprocess.Popen([sys.executable, script_path])
            messagebox.showinfo("Info", "🔍 Face recognition started!\nPress 'q' in camera window to stop.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start face recognition: {e}")

    threading.Thread(target=target, daemon=True).start()

def stop_recognize_face():
    global recognition_process
    if recognition_process:
        try:
            recognition_process.terminate()
            recognition_process.wait(timeout=5)
            recognition_process = None
            messagebox.showinfo("Info", "⏹️ Face recognition stopped.")
        except:
            try:
                recognition_process.kill()
                recognition_process = None
                messagebox.showinfo("Info", "⏹️ Face recognition force stopped.")
            except:
                messagebox.showerror("Error", "Failed to stop recognition.")
    else:
        messagebox.showerror("Error", "Face recognition is not running.")

def shutdown():
    global recognition_process
    if recognition_process:
        try:
            recognition_process.terminate()
        except:
            pass
    root.destroy()

def search_documents():
    year = year_combobox.get()
    month = month_combobox.get()
    day = day_combobox.get()
    file_name = file_name_entry.get()
    
    if not (year and month and day and file_name):
        messagebox.showerror("Error", "Please select year, month, date, and enter a file name.")
        return
    
    # Try multiple file extensions
    extensions = ['.xlsx', '.csv', '.txt']
    found_path = None
    found_name = None
    
    for ext in extensions:
        filename = f"{file_name}_{year}-{month}-{day}{ext}"
        doc_path = get_records_path(filename)
        if os.path.exists(doc_path):
            found_path = doc_path
            found_name = filename
            break
    
    if found_path:
        try:
            os.startfile(found_path)
            messagebox.showinfo("Search Result", f"📂 Opening document: {found_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open document: {str(e)}")
    else:
        messagebox.showerror("Error", 
            f"📄 Document not found!\n\n"
            f"Looked for:\n"
            f"  • {file_name}_{year}-{month}-{day}.xlsx\n"
            f"  • {file_name}_{year}-{month}-{day}.csv\n"
            f"  • {file_name}_{year}-{month}-{day}.txt\n\n"
            f"In: {RECORDS_DIR}"
        )

# ============================================================================
# SPLASH SCREEN
# ============================================================================

def show_splash():
    splash = tk.Toplevel()
    splash.configure(bg='#0a0a0a')
    splash.geometry('400x250')
    splash.overrideredirect(True)
    
    # Center the splash
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() // 2) - 200
    y = (splash.winfo_screenheight() // 2) - 125
    splash.geometry(f'+{x}+{y}')
    
    splash_label = tk.Label(splash, text="🧠 WelCome In Mirror AI ...", 
                            font=("Courier", 18, "bold"), 
                            bg='#0a0a0a', fg='#00ff00')
    splash_label.pack(expand=True)
    
    splash_sub = tk.Label(splash, text="Face Recognition System v3.0", 
                          font=("Courier", 10), 
                          bg='#0a0a0a', fg='#33ff33')
    splash_sub.pack()
    
    # Progress bar
    progress = ttk.Progressbar(splash, length=250, mode='indeterminate')
    progress.pack(pady=20)
    progress.start(15)
    
    # Hide splash after 4 seconds
    root.after(4000, lambda: (progress.stop(), splash.destroy(), show_main_window()))

# ============================================================================
# MAIN WINDOW
# ============================================================================

def show_main_window():
    global root, entry, year_combobox, month_combobox, day_combobox, file_name_entry
    
    root = tk.Tk()
    root.title("Mirror AI - Face Recognition System")
    root.configure(bg='#0a0a0a')
    root.geometry('450x600')
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", shutdown)

    # Sci-Fi Fonts and Colors
    font_large = ("Courier", 14, "bold")
    font_medium = ("Courier", 11)
    font_small = ("Courier", 9)
    color_bg = '#0a0a0a'
    color_fg = '#00ff00'
    color_btn_bg = '#1a1a1a'
    color_btn_fg = '#00ff00'
    color_entry_bg = '#111111'
    color_border = '#1a2a3a'

    # ====================================================================
    # MAIN FRAME
    # ====================================================================

    frame = tk.Frame(root, bg=color_bg)
    frame.pack(padx=20, pady=20)

    # Title
    title = tk.Label(frame, text="⚡ MIRROR AI ⚡", 
                     font=("Courier", 18, "bold"), 
                     bg=color_bg, fg=color_fg)
    title.grid(row=0, column=0, columnspan=2, pady=(0, 15))

    # Name Input
    label = tk.Label(frame, text="Enter your name:", font=font_medium, 
                     bg=color_bg, fg=color_fg)
    label.grid(row=1, column=0, padx=5, pady=5)

    entry = tk.Entry(frame, font=font_medium, 
                     bg=color_entry_bg, fg=color_fg, 
                     insertbackground=color_fg,
                     relief='flat', bd=1)
    entry.grid(row=1, column=1, padx=5, pady=5)

    # Buttons
    capture_button = tk.Button(frame, text="📸 Capture Images", 
                               font=font_medium, 
                               bg=color_btn_bg, fg=color_btn_fg, 
                               relief='flat', bd=1, padx=10, pady=5,
                               cursor='hand2',
                               command=run_capture_images)
    capture_button.grid(row=2, column=0, columnspan=2, pady=5, sticky='ew')

    train_button = tk.Button(frame, text="🎯 Train Model", 
                             font=font_medium, 
                             bg=color_btn_bg, fg=color_btn_fg,
                             relief='flat', bd=1, padx=10, pady=5,
                             cursor='hand2',
                             command=run_train_model)
    train_button.grid(row=3, column=0, columnspan=2, pady=5, sticky='ew')

    recognize_button = tk.Button(frame, text="🔍 Recognize Face", 
                                 font=font_medium, 
                                 bg=color_btn_bg, fg=color_btn_fg,
                                 relief='flat', bd=1, padx=10, pady=5,
                                 cursor='hand2',
                                 command=run_recognize_face)
    recognize_button.grid(row=4, column=0, columnspan=2, pady=5, sticky='ew')

    stop_recognize_button = tk.Button(frame, text="⏹️ Stop Recognition", 
                                      font=font_medium, 
                                      bg='#1a0000', fg='#ff3355',
                                      relief='flat', bd=1, padx=10, pady=5,
                                      cursor='hand2',
                                      command=stop_recognize_face)
    stop_recognize_button.grid(row=5, column=0, columnspan=2, pady=5, sticky='ew')

    # Separator
    separator = tk.Frame(frame, height=2, bg=color_border)
    separator.grid(row=6, column=0, columnspan=2, pady=10, sticky='ew')

    # Off Button
    off_button = tk.Button(frame, text="⏻ Shutdown", 
                           font=font_medium, 
                           bg='#1a0000', fg='#ff3355',
                           relief='flat', bd=1, padx=10, pady=5,
                           cursor='hand2',
                           command=shutdown)
    off_button.grid(row=7, column=0, columnspan=2, pady=5, sticky='ew')

    # ====================================================================
    # SEARCH FRAME
    # ====================================================================

    search_frame = tk.Frame(root, bg=color_bg)
    search_frame.pack(padx=20, pady=(0, 20))

    search_label = tk.Label(search_frame, text="📂 Search Documents", 
                            font=font_large, 
                            bg=color_bg, fg=color_fg)
    search_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

    # Date comboboxes
    years = [str(year) for year in range(2020, datetime.now().year + 1)]
    months = [str(month).zfill(2) for month in range(1, 13)]
    days = [str(day).zfill(2) for day in range(1, 32)]

    year_combobox = ttk.Combobox(search_frame, values=years, 
                                 font=font_small, width=6)
    year_combobox.grid(row=1, column=0, padx=2, pady=5)
    year_combobox.set("Year")

    month_combobox = ttk.Combobox(search_frame, values=months, 
                                  font=font_small, width=4)
    month_combobox.grid(row=1, column=1, padx=2, pady=5)
    month_combobox.set("Month")

    day_combobox = ttk.Combobox(search_frame, values=days, 
                                font=font_small, width=4)
    day_combobox.grid(row=1, column=2, padx=2, pady=5)
    day_combobox.set("Day")

    # File name
    file_name_label = tk.Label(search_frame, text="File Name:", 
                               font=font_medium, 
                               bg=color_bg, fg=color_fg)
    file_name_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')

    file_name_entry = tk.Entry(search_frame, font=font_medium, 
                               bg=color_entry_bg, fg=color_fg, 
                               insertbackground=color_fg,
                               relief='flat', bd=1, width=12)
    file_name_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky='w')
    file_name_entry.insert(0, "attendance")

    # Search button
    search_button = tk.Button(search_frame, text="🔎 Search & Open", 
                              font=font_medium, 
                              bg=color_btn_bg, fg=color_btn_fg,
                              relief='flat', bd=1, padx=15, pady=5,
                              cursor='hand2',
                              command=search_documents)
    search_button.grid(row=3, column=0, columnspan=3, pady=10, sticky='ew')

    # ====================================================================
    # STATUS BAR
    # ====================================================================

    status_frame = tk.Frame(root, bg='#111927', relief='flat', bd=1)
    status_frame.pack(side='bottom', fill='x')

    status_label = tk.Label(status_frame, text="🚀 System Ready", 
                            font=font_small,
                            bg='#111927', fg='#00ff88')
    status_label.pack(side='left', padx=15, pady=5)

    version_label = tk.Label(status_frame, text="v3.0", 
                             font=font_small,
                             bg='#111927', fg='#556677')
    version_label.pack(side='right', padx=15, pady=5)

    # ====================================================================
    # START THE APPLICATION
    # ====================================================================

    root.mainloop()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check for OpenCV
    try:
        import cv2
        print(f"[INFO] OpenCV: {cv2.__version__}")
    except ImportError:
        print("[WARNING] OpenCV not installed")
    
    print(f"[INFO] Scripts Directory: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')}")
    print(f"[INFO] Records Directory: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'records')}")
    
    # Show splash
    root = tk.Tk()
    root.withdraw()
    show_splash()
    root.mainloop()